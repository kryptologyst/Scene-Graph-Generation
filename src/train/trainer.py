"""Training utilities and trainer class."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.structures import SceneGraphBatch, SceneGraphMetrics
from ..utils.device import get_device, get_mixed_precision_dtype
from ..utils.visualization import setup_logging
from .losses import SceneGraphLoss


class SceneGraphTrainer:
    """Trainer class for scene graph generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "auto",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        max_epochs: int = 100,
        gradient_clip_val: float = 1.0,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        object_loss_weight: float = 1.0,
        relationship_loss_weight: float = 1.0,
        attribute_loss_weight: float = 0.5,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_top_k: int = 3,
        monitor: str = "val/mAP@50",
        mode: str = "max",
        early_stopping_patience: int = 10,
        use_mixed_precision: bool = True,
        compile_model: bool = False
    ):
        """Initialize trainer.
        
        Args:
            model: Scene graph model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to use for training.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            max_epochs: Maximum number of epochs.
            gradient_clip_val: Gradient clipping value.
            optimizer: Optimizer type.
            scheduler: Learning rate scheduler type.
            warmup_epochs: Number of warmup epochs.
            object_loss_weight: Weight for object loss.
            relationship_loss_weight: Weight for relationship loss.
            attribute_loss_weight: Weight for attribute loss.
            checkpoint_dir: Directory to save checkpoints.
            log_dir: Directory to save logs.
            save_top_k: Number of best checkpoints to keep.
            monitor: Metric to monitor for checkpointing.
            mode: Whether to maximize or minimize the monitored metric.
            early_stopping_patience: Patience for early stopping.
            use_mixed_precision: Whether to use mixed precision training.
            compile_model: Whether to compile the model.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device(device)
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.early_stopping_patience = early_stopping_patience
        self.use_mixed_precision = use_mixed_precision
        self.compile_model = compile_model
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(log_file=str(self.log_dir / "training.log"))
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile model if requested
        if compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Setup loss function
        self.criterion = SceneGraphLoss(
            object_loss_weight=object_loss_weight,
            relationship_loss_weight=relationship_loss_weight,
            attribute_loss_weight=attribute_loss_weight
        )
        
        # Setup optimizer
        if optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Setup scheduler
        if scheduler.lower() == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs
            )
        elif scheduler.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max_epochs // 3, gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Setup mixed precision
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            self.dtype = get_mixed_precision_dtype(self.device)
        else:
            self.scaler = None
            self.dtype = torch.float32
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.patience_counter = 0
        self.checkpoint_history = []
        
        # Metrics
        self.train_metrics = SceneGraphMetrics(
            num_object_classes=model.num_object_classes,
            num_predicate_classes=model.num_predicate_classes
        )
        self.val_metrics = SceneGraphMetrics(
            num_object_classes=model.num_object_classes,
            num_predicate_classes=model.num_predicate_classes
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(batch.images)
                    losses = self._compute_losses(predictions, batch)
            else:
                predictions = self.model(batch.images)
                losses = self._compute_losses(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
            
            # Update metrics
            self._update_metrics(predictions, batch, self.train_metrics)
            
            # Update progress bar
            total_loss += losses["total_loss"].item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        predictions = self.model(batch.images)
                        losses = self._compute_losses(predictions, batch)
                else:
                    predictions = self.model(batch.images)
                    losses = self._compute_losses(predictions, batch)
                
                # Update metrics
                self._update_metrics(predictions, batch, self.val_metrics)
                
                total_loss += losses["total_loss"].item()
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute()
        epoch_metrics["loss"] = total_loss / num_batches
        
        return epoch_metrics
    
    def _move_batch_to_device(self, batch: SceneGraphBatch) -> SceneGraphBatch:
        """Move batch to device."""
        return SceneGraphBatch(
            images=batch.images.to(self.device),
            object_boxes=batch.object_boxes.to(self.device),
            object_labels=batch.object_labels.to(self.device),
            object_scores=batch.object_scores.to(self.device),
            relationship_triplets=batch.relationship_triplets.to(self.device),
            relationship_scores=batch.relationship_scores.to(self.device),
            valid_objects=batch.valid_objects.to(self.device),
            valid_relationships=batch.valid_relationships.to(self.device)
        )
    
    def _compute_losses(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: SceneGraphBatch
    ) -> Dict[str, torch.Tensor]:
        """Compute losses."""
        # Prepare targets
        targets = {
            "object_labels": batch.object_labels,
            "relationship_labels": batch.relationship_triplets[:, :, 2],  # Predicate labels
            "bbox_targets": batch.object_boxes  # Simplified bbox targets
        }
        
        return self.criterion(
            predictions, targets, batch.valid_objects, batch.valid_relationships
        )
    
    def _update_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: SceneGraphBatch,
        metrics: SceneGraphMetrics
    ):
        """Update metrics."""
        # Get predictions
        pred_objects = predictions["object_logits"].argmax(dim=-1)
        pred_relationships = predictions["relationship_logits"].argmax(dim=-1)
        
        # Get targets
        target_objects = batch.object_labels
        target_relationships = batch.relationship_triplets[:, :, 2]
        
        # Update metrics
        metrics.update(
            pred_objects, pred_relationships,
            target_objects, target_relationships,
            batch.valid_objects, batch.valid_relationships
        )
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if self._should_stop_early(val_metrics):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.logger.info("Training completed!")
    
    def _log_metrics(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Log training metrics."""
        log_msg = f"Epoch {epoch}: "
        
        # Training metrics
        for key, value in train_metrics.items():
            log_msg += f"train/{key}={value:.4f} "
        
        # Validation metrics
        for key, value in val_metrics.items():
            log_msg += f"val/{key}={value:.4f} "
        
        self.logger.info(log_msg)
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric": self.best_metric
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if self.monitor in val_metrics:
            current_metric = val_metrics[self.monitor]
            
            is_better = (
                (self.mode == "max" and current_metric > self.best_metric) or
                (self.mode == "min" and current_metric < self.best_metric)
            )
            
            if is_better:
                self.best_metric = current_metric
                best_checkpoint_path = self.checkpoint_dir / "best.pt"
                torch.save(checkpoint, best_checkpoint_path)
                
                # Update checkpoint history
                self.checkpoint_history.append({
                    "epoch": epoch,
                    "metric": current_metric,
                    "path": str(best_checkpoint_path)
                })
                
                # Keep only top-k checkpoints
                if len(self.checkpoint_history) > self.save_top_k:
                    old_checkpoint = self.checkpoint_history.pop(0)
                    if os.path.exists(old_checkpoint["path"]):
                        os.remove(old_checkpoint["path"])
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early."""
        if self.monitor not in val_metrics:
            return False
        
        current_metric = val_metrics[self.monitor]
        
        is_better = (
            (self.mode == "max" and current_metric > self.best_metric) or
            (self.mode == "min" and current_metric < self.best_metric)
        )
        
        if is_better:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.early_stopping_patience
