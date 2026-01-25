"""Data structures and classes for scene graph generation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class ObjectAnnotation:
    """Annotation for a single object in an image.
    
    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        label: Object class label.
        score: Confidence score.
        attributes: Optional attributes for the object.
    """
    bbox: List[float]
    label: str
    score: float = 1.0
    attributes: Optional[List[str]] = None


@dataclass
class RelationshipAnnotation:
    """Annotation for a relationship between two objects.
    
    Attributes:
        subject_idx: Index of the subject object.
        object_idx: Index of the object.
        predicate: Relationship predicate.
        score: Confidence score.
    """
    subject_idx: int
    object_idx: int
    predicate: str
    score: float = 1.0


@dataclass
class SceneGraphAnnotation:
    """Complete scene graph annotation for an image.
    
    Attributes:
        image_id: Unique identifier for the image.
        image_path: Path to the image file.
        objects: List of object annotations.
        relationships: List of relationship annotations.
        image_size: Image dimensions (width, height).
    """
    image_id: str
    image_path: str
    objects: List[ObjectAnnotation]
    relationships: List[RelationshipAnnotation]
    image_size: Tuple[int, int]


@dataclass
class SceneGraphBatch:
    """Batch of scene graph data for training/inference.
    
    Attributes:
        images: Batch of images [B, C, H, W].
        object_boxes: Object bounding boxes [B, N, 4].
        object_labels: Object class labels [B, N].
        object_scores: Object confidence scores [B, N].
        relationship_triplets: Relationship triplets [B, M, 3] (subj, obj, pred).
        relationship_scores: Relationship confidence scores [B, M].
        valid_objects: Valid object mask [B, N].
        valid_relationships: Valid relationship mask [B, M].
    """
    images: Tensor
    object_boxes: Tensor
    object_labels: Tensor
    object_scores: Tensor
    relationship_triplets: Tensor
    relationship_scores: Tensor
    valid_objects: Tensor
    valid_relationships: Tensor


class SceneGraphMetrics:
    """Metrics for scene graph evaluation."""
    
    def __init__(self, num_object_classes: int, num_predicate_classes: int):
        """Initialize metrics calculator.
        
        Args:
            num_object_classes: Number of object classes.
            num_predicate_classes: Number of predicate classes.
        """
        self.num_object_classes = num_object_classes
        self.num_predicate_classes = num_predicate_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.object_correct = 0
        self.object_total = 0
        self.relationship_correct = 0
        self.relationship_total = 0
        self.triplet_correct = 0
        self.triplet_total = 0
        
        # Per-class metrics
        self.object_class_correct = torch.zeros(self.num_object_classes)
        self.object_class_total = torch.zeros(self.num_object_classes)
        self.predicate_class_correct = torch.zeros(self.num_predicate_classes)
        self.predicate_class_total = torch.zeros(self.num_predicate_classes)
    
    def update(
        self,
        pred_objects: Tensor,
        pred_relationships: Tensor,
        target_objects: Tensor,
        target_relationships: Tensor,
        valid_objects: Tensor,
        valid_relationships: Tensor
    ) -> None:
        """Update metrics with a batch of predictions.
        
        Args:
            pred_objects: Predicted object labels [B, N].
            pred_relationships: Predicted relationship labels [B, M].
            target_objects: Target object labels [B, N].
            target_relationships: Target relationship labels [B, M].
            valid_objects: Valid object mask [B, N].
            valid_relationships: Valid relationship mask [B, M].
        """
        batch_size = pred_objects.size(0)
        
        # Object accuracy
        for b in range(batch_size):
            valid_mask = valid_objects[b]
            if valid_mask.sum() > 0:
                pred_obj = pred_objects[b][valid_mask]
                target_obj = target_objects[b][valid_mask]
                
                self.object_correct += (pred_obj == target_obj).sum().item()
                self.object_total += valid_mask.sum().item()
                
                # Per-class object accuracy
                for class_id in range(self.num_object_classes):
                    class_mask = target_obj == class_id
                    if class_mask.sum() > 0:
                        self.object_class_correct[class_id] += (pred_obj[class_mask] == class_id).sum().item()
                        self.object_class_total[class_id] += class_mask.sum().item()
        
        # Relationship accuracy
        for b in range(batch_size):
            valid_mask = valid_relationships[b]
            if valid_mask.sum() > 0:
                pred_rel = pred_relationships[b][valid_mask]
                target_rel = target_relationships[b][valid_mask]
                
                self.relationship_correct += (pred_rel == target_rel).sum().item()
                self.relationship_total += valid_mask.sum().item()
                
                # Per-class predicate accuracy
                for class_id in range(self.num_predicate_classes):
                    class_mask = target_rel == class_id
                    if class_mask.sum() > 0:
                        self.predicate_class_correct[class_id] += (pred_rel[class_mask] == class_id).sum().item()
                        self.predicate_class_total[class_id] += class_mask.sum().item()
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        metrics = {}
        
        # Overall accuracy
        if self.object_total > 0:
            metrics["object_accuracy"] = self.object_correct / self.object_total
        else:
            metrics["object_accuracy"] = 0.0
        
        if self.relationship_total > 0:
            metrics["relationship_accuracy"] = self.relationship_correct / self.relationship_total
        else:
            metrics["relationship_accuracy"] = 0.0
        
        # Per-class accuracy
        valid_object_classes = self.object_class_total > 0
        if valid_object_classes.sum() > 0:
            metrics["mean_object_class_accuracy"] = (
                self.object_class_correct[valid_object_classes] / 
                self.object_class_total[valid_object_classes]
            ).mean().item()
        else:
            metrics["mean_object_class_accuracy"] = 0.0
        
        valid_predicate_classes = self.predicate_class_total > 0
        if valid_predicate_classes.sum() > 0:
            metrics["mean_predicate_class_accuracy"] = (
                self.predicate_class_correct[valid_predicate_classes] / 
                self.predicate_class_total[valid_predicate_classes]
            ).mean().item()
        else:
            metrics["mean_predicate_class_accuracy"] = 0.0
        
        return metrics
