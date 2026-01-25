"""Scene graph generation models."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from .layers import GraphConvolution, ObjectDetector, PositionalEncoding, RelationAttention, RelationshipHead


class MotifNet(nn.Module):
    """MotifNet model for scene graph generation.
    
    This model implements the MotifNet architecture which uses a CNN backbone
    for object detection and a graph neural network for relationship prediction.
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        num_object_classes: int = 150,
        num_predicate_classes: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_fpn: bool = True,
        roi_pool_size: int = 7,
        num_proposals: int = 100,
        nms_threshold: float = 0.7,
        score_threshold: float = 0.05,
        use_context: bool = True,
        use_attention: bool = True,
        attention_heads: int = 8,
        max_objects: int = 50,
        max_relationships: int = 100
    ):
        """Initialize MotifNet model.
        
        Args:
            backbone: CNN backbone architecture.
            num_object_classes: Number of object classes.
            num_predicate_classes: Number of predicate classes.
            hidden_dim: Hidden dimension for features.
            num_layers: Number of graph convolution layers.
            dropout: Dropout rate.
            use_fpn: Whether to use Feature Pyramid Network.
            roi_pool_size: ROI pooling size.
            num_proposals: Number of object proposals.
            nms_threshold: NMS threshold for object detection.
            score_threshold: Score threshold for object detection.
            use_context: Whether to use contextual information.
            use_attention: Whether to use attention mechanism.
            attention_heads: Number of attention heads.
            max_objects: Maximum number of objects per image.
            max_relationships: Maximum number of relationships per image.
        """
        super().__init__()
        
        self.num_object_classes = num_object_classes
        self.num_predicate_classes = num_predicate_classes
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_objects = max_objects
        self.max_relationships = max_relationships
        
        # Backbone network
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature Pyramid Network (optional)
        if use_fpn:
            self.fpn = nn.ModuleList([
                nn.Conv2d(backbone_dim, hidden_dim, 1),
                nn.Conv2d(backbone_dim, hidden_dim, 1),
                nn.Conv2d(backbone_dim, hidden_dim, 1)
            ])
            feature_dim = hidden_dim
        else:
            self.fpn = None
            feature_dim = backbone_dim
        
        # Object detector
        self.object_detector = ObjectDetector(
            feature_dim, num_object_classes, roi_pool_size, hidden_dim
        )
        
        # Object feature projection
        self.object_proj = nn.Linear(feature_dim * roi_pool_size * roi_pool_size, hidden_dim)
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = RelationAttention(hidden_dim, attention_heads, dropout)
        else:
            self.attention = None
        
        # Relationship head
        self.relationship_head = RelationshipHead(
            hidden_dim, num_predicate_classes, hidden_dim, use_context
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.
        
        Args:
            images: Input images [batch_size, 3, height, width].
            
        Returns:
            Feature maps [batch_size, channels, height, width].
        """
        features = self.backbone(images)
        
        if self.fpn is not None:
            # Simple FPN implementation
            features = self.fpn[0](features)
        
        return features
    
    def detect_objects(
        self, 
        features: torch.Tensor, 
        rois: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect objects in the image.
        
        Args:
            features: Feature maps [batch_size, channels, height, width].
            rois: Optional region proposals [batch_size, num_rois, 4].
            
        Returns:
            Tuple of (object_logits, bbox_deltas, object_features).
        """
        if rois is None:
            # Generate default ROIs (sliding window)
            batch_size, _, height, width = features.size()
            rois = self._generate_default_rois(batch_size, height, width)
        
        # Object detection
        object_logits, bbox_deltas = self.object_detector(features, rois)
        
        # Extract object features
        batch_size, num_rois = rois.size(0), rois.size(1)
        object_features = []
        
        for b in range(batch_size):
            batch_features = []
            for r in range(num_rois):
                roi = rois[b, r]
                x1, y1, x2, y2 = roi.int()
                
                roi_feat = features[b, :, y1:y2+1, x1:x2+1]
                roi_feat = torch.nn.functional.adaptive_avg_pool2d(roi_feat, (7, 7))
                batch_features.append(roi_feat.flatten())
            
            object_features.append(torch.stack(batch_features))
        
        object_features = torch.stack(object_features)
        object_features = self.object_proj(object_features)
        
        return object_logits, bbox_deltas, object_features
    
    def _generate_default_rois(self, batch_size: int, height: int, width: int) -> torch.Tensor:
        """Generate default region proposals.
        
        Args:
            batch_size: Batch size.
            height: Feature map height.
            width: Feature map width.
            
        Returns:
            Region proposals [batch_size, num_proposals, 4].
        """
        rois = []
        
        # Generate ROIs using sliding window
        stride = 16
        roi_size = 32
        
        for y in range(0, height - roi_size + 1, stride):
            for x in range(0, width - roi_size + 1, stride):
                rois.append([x, y, x + roi_size, y + roi_size])
        
        # Limit to num_proposals
        rois = rois[:self.num_proposals]
        
        # Pad if necessary
        while len(rois) < self.num_proposals:
            rois.append([0, 0, roi_size, roi_size])
        
        rois = torch.tensor(rois, dtype=torch.float32)
        rois = rois.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return rois
    
    def predict_relationships(
        self,
        object_features: torch.Tensor,
        object_logits: torch.Tensor,
        valid_objects: torch.Tensor
    ) -> torch.Tensor:
        """Predict relationships between objects.
        
        Args:
            object_features: Object features [batch_size, num_objects, feature_dim].
            object_logits: Object classification logits [batch_size, num_objects, num_classes].
            valid_objects: Valid object mask [batch_size, num_objects].
            
        Returns:
            Relationship logits [batch_size, num_relationships, num_predicates].
        """
        batch_size, num_objects = object_features.size(0), object_features.size(1)
        
        # Generate all possible object pairs
        subject_indices = []
        object_indices = []
        
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:  # No self-relationships
                    subject_indices.append(i)
                    object_indices.append(j)
        
        subject_indices = torch.tensor(subject_indices, device=object_features.device)
        object_indices = torch.tensor(object_indices, device=object_features.device)
        
        # Limit to max_relationships
        if len(subject_indices) > self.max_relationships:
            indices = torch.randperm(len(subject_indices))[:self.max_relationships]
            subject_indices = subject_indices[indices]
            object_indices = object_indices[indices]
        
        # Expand to batch dimension
        subject_indices = subject_indices.unsqueeze(0).repeat(batch_size, 1)
        object_indices = object_indices.unsqueeze(0).repeat(batch_size, 1)
        
        # Apply graph convolution layers
        graph_features = object_features
        for graph_layer in self.graph_layers:
            # Create adjacency matrix (simplified - all objects connected)
            adj = torch.ones(batch_size, num_objects, num_objects, device=object_features.device)
            adj = adj / adj.sum(dim=-1, keepdim=True)  # Normalize
            
            graph_features = graph_layer(graph_features, adj)
            graph_features = torch.relu(graph_features)
            graph_features = self.dropout(graph_features)
        
        # Apply attention (optional)
        if self.attention is not None:
            graph_features = self.attention(graph_features, graph_features, graph_features)
        
        # Predict relationships
        relationship_logits = self.relationship_head(
            graph_features, subject_indices, object_indices
        )
        
        return relationship_logits
    
    def forward(
        self, 
        images: torch.Tensor,
        rois: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            images: Input images [batch_size, 3, height, width].
            rois: Optional region proposals.
            
        Returns:
            Dictionary containing predictions.
        """
        # Extract features
        features = self.extract_features(images)
        
        # Detect objects
        object_logits, bbox_deltas, object_features = self.detect_objects(features, rois)
        
        # Create valid object mask (simplified)
        batch_size, num_objects = object_logits.size(0), object_logits.size(1)
        valid_objects = torch.ones(batch_size, num_objects, dtype=torch.bool, device=images.device)
        
        # Predict relationships
        relationship_logits = self.predict_relationships(
            object_features, object_logits, valid_objects
        )
        
        return {
            "object_logits": object_logits,
            "bbox_deltas": bbox_deltas,
            "object_features": object_features,
            "relationship_logits": relationship_logits,
            "valid_objects": valid_objects
        }
