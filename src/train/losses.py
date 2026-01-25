"""Loss functions for scene graph generation."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGraphLoss(nn.Module):
    """Combined loss function for scene graph generation.
    
    This loss combines object detection loss and relationship prediction loss.
    """
    
    def __init__(
        self,
        object_loss_weight: float = 1.0,
        relationship_loss_weight: float = 1.0,
        attribute_loss_weight: float = 0.5,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        """Initialize scene graph loss.
        
        Args:
            object_loss_weight: Weight for object detection loss.
            relationship_loss_weight: Weight for relationship loss.
            attribute_loss_weight: Weight for attribute loss.
            use_focal_loss: Whether to use focal loss for classification.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        
        self.object_loss_weight = object_loss_weight
        self.relationship_loss_weight = relationship_loss_weight
        self.attribute_loss_weight = attribute_loss_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        
        # Loss functions
        if use_focal_loss:
            self.object_loss_fn = FocalLoss(focal_alpha, focal_gamma)
            self.relationship_loss_fn = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.object_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.relationship_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.bbox_loss_fn = nn.SmoothL1Loss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_objects: torch.Tensor,
        valid_relationships: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute scene graph loss.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            valid_objects: Valid object mask.
            valid_relationships: Valid relationship mask.
            
        Returns:
            Dictionary of loss components.
        """
        losses = {}
        
        # Object detection loss
        if "object_logits" in predictions and "object_labels" in targets:
            object_logits = predictions["object_logits"]
            object_labels = targets["object_labels"]
            
            # Apply valid mask
            valid_mask = valid_objects.unsqueeze(-1).expand_as(object_logits)
            object_logits_masked = object_logits[valid_mask].view(-1, object_logits.size(-1))
            object_labels_masked = object_labels[valid_objects]
            
            if len(object_labels_masked) > 0:
                losses["object_loss"] = self.object_loss_fn(object_logits_masked, object_labels_masked)
            else:
                losses["object_loss"] = torch.tensor(0.0, device=object_logits.device)
        
        # Relationship loss
        if "relationship_logits" in predictions and "relationship_labels" in targets:
            relationship_logits = predictions["relationship_logits"]
            relationship_labels = targets["relationship_labels"]
            
            # Apply valid mask
            valid_mask = valid_relationships.unsqueeze(-1).expand_as(relationship_logits)
            relationship_logits_masked = relationship_logits[valid_mask].view(-1, relationship_logits.size(-1))
            relationship_labels_masked = relationship_labels[valid_relationships]
            
            if len(relationship_labels_masked) > 0:
                losses["relationship_loss"] = self.relationship_loss_fn(
                    relationship_logits_masked, relationship_labels_masked
                )
            else:
                losses["relationship_loss"] = torch.tensor(0.0, device=relationship_logits.device)
        
        # Bounding box regression loss
        if "bbox_deltas" in predictions and "bbox_targets" in targets:
            bbox_deltas = predictions["bbox_deltas"]
            bbox_targets = targets["bbox_targets"]
            
            # Apply valid mask
            valid_mask = valid_objects.unsqueeze(-1).expand_as(bbox_deltas)
            bbox_deltas_masked = bbox_deltas[valid_mask].view(-1, bbox_deltas.size(-1))
            bbox_targets_masked = bbox_targets[valid_objects]
            
            if len(bbox_targets_masked) > 0:
                losses["bbox_loss"] = self.bbox_loss_fn(bbox_deltas_masked, bbox_targets_masked)
            else:
                losses["bbox_loss"] = torch.tensor(0.0, device=bbox_deltas.device)
        
        # Total loss
        total_loss = (
            self.object_loss_weight * losses.get("object_loss", 0.0) +
            self.relationship_loss_weight * losses.get("relationship_loss", 0.0) +
            self.attribute_loss_weight * losses.get("bbox_loss", 0.0)
        )
        
        losses["total_loss"] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss is designed to down-weight easy examples and focus on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted logits [N, C].
            targets: Target class indices [N].
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class RelationshipLoss(nn.Module):
    """Specialized loss for relationship prediction.
    
    This loss considers the structured nature of relationships and can incorporate
    additional constraints like relationship consistency.
    """
    
    def __init__(
        self,
        num_predicates: int,
        use_consistency_loss: bool = True,
        consistency_weight: float = 0.1
    ):
        """Initialize relationship loss.
        
        Args:
            num_predicates: Number of predicate classes.
            use_consistency_loss: Whether to use consistency loss.
            consistency_weight: Weight for consistency loss.
        """
        super().__init__()
        self.num_predicates = num_predicates
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        
        if use_consistency_loss:
            self.consistency_loss_fn = nn.MSELoss()
    
    def forward(
        self,
        relationship_logits: torch.Tensor,
        relationship_labels: torch.Tensor,
        subject_features: torch.Tensor,
        object_features: torch.Tensor,
        valid_relationships: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute relationship loss.
        
        Args:
            relationship_logits: Predicted relationship logits.
            relationship_labels: Ground truth relationship labels.
            subject_features: Subject object features.
            object_features: Object features.
            valid_relationships: Valid relationship mask.
            
        Returns:
            Dictionary of loss components.
        """
        losses = {}
        
        # Classification loss
        valid_mask = valid_relationships.unsqueeze(-1).expand_as(relationship_logits)
        logits_masked = relationship_logits[valid_mask].view(-1, relationship_logits.size(-1))
        labels_masked = relationship_labels[valid_relationships]
        
        if len(labels_masked) > 0:
            losses["classification_loss"] = self.classification_loss(logits_masked, labels_masked)
        else:
            losses["classification_loss"] = torch.tensor(0.0, device=relationship_logits.device)
        
        # Consistency loss (optional)
        if self.use_consistency_loss:
            # Encourage symmetric relationships to have similar features
            consistency_loss = self._compute_consistency_loss(
                subject_features, object_features, valid_relationships
            )
            losses["consistency_loss"] = consistency_loss
        
        return losses
    
    def _compute_consistency_loss(
        self,
        subject_features: torch.Tensor,
        object_features: torch.Tensor,
        valid_relationships: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency loss for relationships.
        
        Args:
            subject_features: Subject object features.
            object_features: Object features.
            valid_relationships: Valid relationship mask.
            
        Returns:
            Consistency loss value.
        """
        # Simple consistency: encourage similar features for symmetric relationships
        batch_size, num_objects, feature_dim = subject_features.size()
        
        # Compute feature similarity matrix
        similarity_matrix = torch.matmul(
            subject_features, object_features.transpose(-2, -1)
        )
        
        # Apply valid mask
        valid_mask = valid_relationships.unsqueeze(-1) & valid_relationships.unsqueeze(-2)
        
        # Consistency loss: encourage symmetric relationships
        consistency_loss = torch.mean(similarity_matrix[valid_mask])
        
        return consistency_loss
