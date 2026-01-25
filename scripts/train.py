#!/usr/bin/env python3
"""Main training script for scene graph generation."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.datasets import VisualGenomeDataset, collate_fn
from src.models.scene_graph import MotifNet
from src.train.trainer import SceneGraphTrainer
from src.utils.device import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train scene graph generation model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Log directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        # Return default config
        return {
            "project_name": "scene_graph_generation",
            "experiment_name": "sg_gen_experiment",
            "seed": 42,
            "data_dir": "data",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
            "device": "auto",
            "mixed_precision": True,
            "compile_model": False,
            "log_level": "INFO",
            "use_wandb": False,
            "wandb_project": "scene_graph_generation",
            "eval_interval": 1,
            "save_interval": 5,
            "max_checkpoints": 3,
            "model": {
                "backbone": "resnet50",
                "num_object_classes": 150,
                "num_predicate_classes": 50,
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.1,
                "use_fpn": True,
                "roi_pool_size": 7,
                "num_proposals": 100,
                "nms_threshold": 0.7,
                "score_threshold": 0.05,
                "use_context": True,
                "use_attention": True,
                "attention_heads": 8,
                "max_objects": 50,
                "max_relationships": 100
            },
            "data": {
                "data_dir": "data",
                "image_dir": "data/raw/images",
                "annotation_file": "data/raw/annotations.json",
                "batch_size": 8,
                "num_workers": 4,
                "pin_memory": True,
                "persistent_workers": True,
                "image_size": [512, 512],
                "min_image_size": 224,
                "max_image_size": 1024,
                "use_augmentation": True,
                "horizontal_flip": 0.5,
                "color_jitter": 0.1,
                "rotation": 10,
                "scale": [0.8, 1.2],
                "max_objects_per_image": 50,
                "max_relationships_per_image": 100,
                "min_object_area": 0.001,
                "min_relationship_area": 0.0001
            },
            "trainer": {
                "max_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_val": 1.0,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_epochs": 5,
                "object_loss_weight": 1.0,
                "relationship_loss_weight": 1.0,
                "attribute_loss_weight": 0.5,
                "val_check_interval": 1.0,
                "early_stopping_patience": 10,
                "monitor": "val/mAP@50",
                "save_top_k": 3,
                "mode": "max",
                "save_last": True
            }
        }
    
    return OmegaConf.load(config_path)


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Create data loaders for training and validation."""
    data_config = config["data"]
    
    # Create datasets
    train_dataset = VisualGenomeDataset(
        data_dir=data_config["data_dir"],
        image_dir=data_config["image_dir"],
        annotation_file=data_config["annotation_file"],
        image_size=tuple(data_config["image_size"]),
        min_image_size=data_config["min_image_size"],
        max_image_size=data_config["max_image_size"],
        use_augmentation=data_config["use_augmentation"],
        horizontal_flip=data_config["horizontal_flip"],
        color_jitter=data_config["color_jitter"],
        rotation=data_config["rotation"],
        scale=tuple(data_config["scale"]),
        max_objects_per_image=data_config["max_objects_per_image"],
        max_relationships_per_image=data_config["max_relationships_per_image"],
        min_object_area=data_config["min_object_area"],
        min_relationship_area=data_config["min_relationship_area"],
        split="train"
    )
    
    val_dataset = VisualGenomeDataset(
        data_dir=data_config["data_dir"],
        image_dir=data_config["image_dir"],
        annotation_file=data_config["annotation_file"],
        image_size=tuple(data_config["image_size"]),
        min_image_size=data_config["min_image_size"],
        max_image_size=data_config["max_image_size"],
        use_augmentation=False,  # No augmentation for validation
        split="val"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        persistent_workers=data_config["persistent_workers"],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
        persistent_workers=data_config["persistent_workers"],
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def create_model(config: Dict[str, Any]) -> MotifNet:
    """Create scene graph model."""
    model_config = config["model"]
    
    model = MotifNet(
        backbone=model_config["backbone"],
        num_object_classes=model_config["num_object_classes"],
        num_predicate_classes=model_config["num_predicate_classes"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        use_fpn=model_config["use_fpn"],
        roi_pool_size=model_config["roi_pool_size"],
        num_proposals=model_config["num_proposals"],
        nms_threshold=model_config["nms_threshold"],
        score_threshold=model_config["score_threshold"],
        use_context=model_config["use_context"],
        use_attention=model_config["use_attention"],
        attention_heads=model_config["attention_heads"],
        max_objects=model_config["max_objects"],
        max_relationships=model_config["max_relationships"]
    )
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    config["data_dir"] = args.data_dir
    config["checkpoint_dir"] = args.checkpoint_dir
    config["log_dir"] = args.log_dir
    config["device"] = args.device
    config["seed"] = args.seed
    
    # Set random seed
    set_seed(config["seed"])
    
    # Create directories
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Print model info
    from src.utils.device import count_parameters, get_model_size_mb
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Create trainer
    trainer_config = config["trainer"]
    trainer = SceneGraphTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        learning_rate=trainer_config["learning_rate"],
        weight_decay=trainer_config["weight_decay"],
        max_epochs=trainer_config["max_epochs"],
        gradient_clip_val=trainer_config["gradient_clip_val"],
        optimizer=trainer_config["optimizer"],
        scheduler=trainer_config["scheduler"],
        warmup_epochs=trainer_config["warmup_epochs"],
        object_loss_weight=trainer_config["object_loss_weight"],
        relationship_loss_weight=trainer_config["relationship_loss_weight"],
        attribute_loss_weight=trainer_config["attribute_loss_weight"],
        checkpoint_dir=config["checkpoint_dir"],
        log_dir=config["log_dir"],
        save_top_k=trainer_config["save_top_k"],
        monitor=trainer_config["monitor"],
        mode=trainer_config["mode"],
        early_stopping_patience=trainer_config["early_stopping_patience"],
        use_mixed_precision=config["mixed_precision"],
        compile_model=config["compile_model"]
    )
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_metric = checkpoint["best_metric"]
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
