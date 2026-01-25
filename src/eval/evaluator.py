"""Evaluation utilities for scene graph generation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.structures import SceneGraphBatch, SceneGraphMetrics
from ..utils.device import get_device
from ..utils.visualization import save_results_json, create_metrics_table


class SceneGraphEvaluator:
    """Evaluator for scene graph generation models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "auto",
        output_dir: str = "evaluation_results",
        class_names: Optional[Tuple[List[str], List[str]]] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained scene graph model.
            test_loader: Test data loader.
            device: Device to use for evaluation.
            output_dir: Directory to save evaluation results.
            class_names: Tuple of (object_class_names, predicate_class_names).
        """
        self.model = model
        self.test_loader = test_loader
        self.device = get_device(device)
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.metrics = SceneGraphMetrics(
            num_object_classes=model.num_object_classes,
            num_predicate_classes=model.num_predicate_classes
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation."""
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                predictions = self.model(batch.images)
                
                # Store predictions and targets
                batch_predictions = self._extract_predictions(predictions, batch)
                batch_targets = self._extract_targets(batch)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                all_image_ids.extend(batch.image_id if hasattr(batch, 'image_id') else [])
                
                # Update metrics
                self._update_metrics(predictions, batch)
        
        # Compute final metrics
        final_metrics = self.metrics.compute()
        
        # Compute additional metrics
        additional_metrics = self._compute_additional_metrics(all_predictions, all_targets)
        final_metrics.update(additional_metrics)
        
        # Save results
        results = {
            "metrics": final_metrics,
            "predictions": all_predictions,
            "targets": all_targets,
            "image_ids": all_image_ids
        }
        
        self._save_results(results)
        
        return results
    
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
    
    def _extract_predictions(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: SceneGraphBatch
    ) -> List[Dict[str, Any]]:
        """Extract predictions from model output."""
        batch_predictions = []
        
        batch_size = batch.images.size(0)
        
        for b in range(batch_size):
            # Object predictions
            object_logits = predictions["object_logits"][b]
            object_probs = torch.softmax(object_logits, dim=-1)
            object_preds = object_logits.argmax(dim=-1)
            
            # Relationship predictions
            relationship_logits = predictions["relationship_logits"][b]
            relationship_probs = torch.softmax(relationship_logits, dim=-1)
            relationship_preds = relationship_logits.argmax(dim=-1)
            
            # Apply valid masks
            valid_objects = batch.valid_objects[b]
            valid_relationships = batch.valid_relationships[b]
            
            # Extract valid predictions
            valid_object_preds = object_preds[valid_objects]
            valid_object_probs = object_probs[valid_objects]
            valid_relationship_preds = relationship_preds[valid_relationships]
            valid_relationship_probs = relationship_probs[valid_relationships]
            
            batch_predictions.append({
                "objects": {
                    "predictions": valid_object_preds.cpu().numpy(),
                    "probabilities": valid_object_probs.cpu().numpy(),
                    "boxes": batch.object_boxes[b][valid_objects].cpu().numpy()
                },
                "relationships": {
                    "predictions": valid_relationship_preds.cpu().numpy(),
                    "probabilities": valid_relationship_probs.cpu().numpy(),
                    "triplets": batch.relationship_triplets[b][valid_relationships].cpu().numpy()
                }
            })
        
        return batch_predictions
    
    def _extract_targets(self, batch: SceneGraphBatch) -> List[Dict[str, Any]]:
        """Extract ground truth targets."""
        batch_targets = []
        
        batch_size = batch.images.size(0)
        
        for b in range(batch_size):
            # Apply valid masks
            valid_objects = batch.valid_objects[b]
            valid_relationships = batch.valid_relationships[b]
            
            batch_targets.append({
                "objects": {
                    "labels": batch.object_labels[b][valid_objects].cpu().numpy(),
                    "boxes": batch.object_boxes[b][valid_objects].cpu().numpy()
                },
                "relationships": {
                    "labels": batch.relationship_triplets[b][valid_relationships][:, 2].cpu().numpy(),
                    "triplets": batch.relationship_triplets[b][valid_relationships].cpu().numpy()
                }
            })
        
        return batch_targets
    
    def _update_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: SceneGraphBatch
    ):
        """Update evaluation metrics."""
        # Get predictions
        pred_objects = predictions["object_logits"].argmax(dim=-1)
        pred_relationships = predictions["relationship_logits"].argmax(dim=-1)
        
        # Get targets
        target_objects = batch.object_labels
        target_relationships = batch.relationship_triplets[:, :, 2]
        
        # Update metrics
        self.metrics.update(
            pred_objects, pred_relationships,
            target_objects, target_relationships,
            batch.valid_objects, batch.valid_relationships
        )
    
    def _compute_additional_metrics(
        self, 
        predictions: List[Dict[str, Any]], 
        targets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute additional evaluation metrics."""
        metrics = {}
        
        # Object detection metrics
        object_precisions = []
        object_recalls = []
        object_f1_scores = []
        
        for pred, target in zip(predictions, targets):
            pred_labels = pred["objects"]["predictions"]
            target_labels = target["objects"]["labels"]
            
            if len(target_labels) > 0:
                # Compute precision, recall, F1
                tp = np.sum(pred_labels == target_labels)
                fp = len(pred_labels) - tp
                fn = len(target_labels) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                object_precisions.append(precision)
                object_recalls.append(recall)
                object_f1_scores.append(f1)
        
        if object_precisions:
            metrics["object_precision"] = np.mean(object_precisions)
            metrics["object_recall"] = np.mean(object_recalls)
            metrics["object_f1"] = np.mean(object_f1_scores)
        
        # Relationship metrics
        relationship_precisions = []
        relationship_recalls = []
        relationship_f1_scores = []
        
        for pred, target in zip(predictions, targets):
            pred_labels = pred["relationships"]["predictions"]
            target_labels = target["relationships"]["labels"]
            
            if len(target_labels) > 0:
                # Compute precision, recall, F1
                tp = np.sum(pred_labels == target_labels)
                fp = len(pred_labels) - tp
                fn = len(target_labels) - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                relationship_precisions.append(precision)
                relationship_recalls.append(recall)
                relationship_f1_scores.append(f1)
        
        if relationship_precisions:
            metrics["relationship_precision"] = np.mean(relationship_precisions)
            metrics["relationship_recall"] = np.mean(relationship_recalls)
            metrics["relationship_f1"] = np.mean(relationship_f1_scores)
        
        # Scene graph completeness metrics
        completeness_scores = []
        for pred, target in zip(predictions, targets):
            pred_objects = len(pred["objects"]["predictions"])
            target_objects = len(target["objects"]["labels"])
            pred_relationships = len(pred["relationships"]["predictions"])
            target_relationships = len(target["relationships"]["labels"])
            
            # Object completeness
            obj_completeness = pred_objects / target_objects if target_objects > 0 else 0.0
            
            # Relationship completeness
            rel_completeness = pred_relationships / target_relationships if target_relationships > 0 else 0.0
            
            # Overall completeness
            completeness = (obj_completeness + rel_completeness) / 2.0
            completeness_scores.append(completeness)
        
        if completeness_scores:
            metrics["scene_graph_completeness"] = np.mean(completeness_scores)
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results."""
        # Save metrics as JSON
        metrics_path = self.output_dir / "metrics.json"
        save_results_json(results["metrics"], str(metrics_path))
        
        # Create metrics table
        table_path = self.output_dir / "metrics_table.png"
        create_metrics_table(results["metrics"], str(table_path))
        
        # Save detailed results
        detailed_path = self.output_dir / "detailed_results.json"
        save_results_json(results, str(detailed_path))
        
        print(f"Evaluation results saved to {self.output_dir}")
        print(f"Metrics: {results['metrics']}")


def compute_recall_at_k(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Compute Recall@K metrics for scene graph generation.
    
    Args:
        predictions: List of predictions.
        targets: List of ground truth targets.
        k_values: List of K values to compute recall for.
        
    Returns:
        Dictionary of Recall@K metrics.
    """
    metrics = {}
    
    for k in k_values:
        recalls = []
        
        for pred, target in zip(predictions, targets):
            # Object recall@k
            pred_objects = pred["objects"]["predictions"]
            target_objects = target["objects"]["labels"]
            
            if len(target_objects) > 0:
                # For simplicity, we'll use exact match
                # In practice, you might want to use IoU-based matching
                correct = np.sum(pred_objects[:k] == target_objects[:k])
                recall = correct / min(k, len(target_objects))
                recalls.append(recall)
        
        if recalls:
            metrics[f"recall_at_{k}"] = np.mean(recalls)
    
    return metrics


def compute_mean_average_precision(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_thresholds: List[float] = [0.5, 0.75, 0.9]
) -> Dict[str, float]:
    """Compute mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of predictions.
        targets: List of ground truth targets.
        iou_thresholds: List of IoU thresholds.
        
    Returns:
        Dictionary of mAP metrics.
    """
    metrics = {}
    
    for threshold in iou_thresholds:
        aps = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred["objects"]["boxes"]
            pred_scores = pred["objects"]["probabilities"].max(axis=1)
            pred_labels = pred["objects"]["predictions"]
            
            target_boxes = target["objects"]["boxes"]
            target_labels = target["objects"]["labels"]
            
            if len(target_boxes) > 0 and len(pred_boxes) > 0:
                # Compute AP for this image
                ap = compute_average_precision(
                    pred_boxes, pred_scores, pred_labels,
                    target_boxes, target_labels, threshold
                )
                aps.append(ap)
        
        if aps:
            metrics[f"mAP@{threshold}"] = np.mean(aps)
    
    return metrics


def compute_average_precision(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    target_boxes: np.ndarray,
    target_labels: np.ndarray,
    iou_threshold: float
) -> float:
    """Compute Average Precision for a single image.
    
    Args:
        pred_boxes: Predicted bounding boxes [N, 4].
        pred_scores: Prediction scores [N].
        pred_labels: Predicted labels [N].
        target_boxes: Ground truth bounding boxes [M, 4].
        target_labels: Ground truth labels [M].
        iou_threshold: IoU threshold for positive detection.
        
    Returns:
        Average Precision value.
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    
    # Compute IoU matrix
    ious = compute_iou_matrix(pred_boxes, target_boxes)
    
    # Match predictions to ground truth
    matched = np.zeros(len(target_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
            if pred_label == target_label and not matched[j]:
                iou = ious[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(target_boxes)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: First set of boxes [N, 4].
        boxes2: Second set of boxes [M, 4].
        
    Returns:
        IoU matrix [N, M].
    """
    # Compute intersection
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute union
    union = area1[:, np.newaxis] + area2 - intersection
    
    # Compute IoU
    iou = intersection / union
    
    return iou
