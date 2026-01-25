"""Core neural network layers for scene graph generation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model].
            
        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:x.size(0), :]


class GraphConvolution(nn.Module):
    """Graph convolution layer for scene graph modeling."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize graph convolution layer.
        
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            bias: Whether to use bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [batch_size, num_nodes, in_features].
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes].
            
        Returns:
            Output features [batch_size, num_nodes, out_features].
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RelationAttention(nn.Module):
    """Attention mechanism for relationship modeling."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize relation attention.
        
        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.attention = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model].
            key: Key tensor [batch_size, seq_len, d_model].
            value: Value tensor [batch_size, seq_len, d_model].
            mask: Attention mask.
            
        Returns:
            Attended features.
        """
        # Multi-head attention
        attn_output, _ = self.attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            key_padding_mask=mask
        )
        
        # Residual connection and layer norm
        output = self.norm(query + attn_output.transpose(0, 1))
        return self.dropout(output)


class ObjectDetector(nn.Module):
    """Object detection head for scene graph generation."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        roi_pool_size: int = 7,
        hidden_dim: int = 256
    ):
        """Initialize object detector.
        
        Args:
            in_channels: Input feature channels.
            num_classes: Number of object classes.
            roi_pool_size: ROI pooling size.
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        
        self.roi_pool_size = roi_pool_size
        self.hidden_dim = hidden_dim
        
        # ROI pooling
        self.roi_pool = nn.AdaptiveAvgPool2d(roi_pool_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * roi_pool_size * roi_pool_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(in_channels * roi_pool_size * roi_pool_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        rois: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: Feature maps [batch_size, channels, height, width].
            rois: Region of interest boxes [batch_size, num_rois, 4].
            
        Returns:
            Tuple of (class_logits, bbox_deltas).
        """
        batch_size, num_rois = rois.size(0), rois.size(1)
        
        # ROI pooling
        pooled_features = []
        for b in range(batch_size):
            batch_features = []
            for r in range(num_rois):
                roi = rois[b, r]
                x1, y1, x2, y2 = roi.int()
                
                # Extract ROI features
                roi_feat = features[b, :, y1:y2+1, x1:x2+1]
                roi_feat = self.roi_pool(roi_feat)
                batch_features.append(roi_feat.flatten())
            
            pooled_features.append(torch.stack(batch_features))
        
        pooled_features = torch.stack(pooled_features)  # [B, N, C*H*W]
        
        # Classification
        class_logits = self.classifier(pooled_features)
        
        # Bounding box regression
        bbox_deltas = self.bbox_regressor(pooled_features)
        
        return class_logits, bbox_deltas


class RelationshipHead(nn.Module):
    """Relationship prediction head for scene graph generation."""
    
    def __init__(
        self,
        object_feature_dim: int,
        num_predicates: int,
        hidden_dim: int = 256,
        use_context: bool = True
    ):
        """Initialize relationship head.
        
        Args:
            object_feature_dim: Object feature dimension.
            num_predicates: Number of predicate classes.
            hidden_dim: Hidden dimension.
            use_context: Whether to use contextual information.
        """
        super().__init__()
        
        self.use_context = use_context
        
        # Subject and object feature processing
        self.subject_proj = nn.Linear(object_feature_dim, hidden_dim)
        self.object_proj = nn.Linear(object_feature_dim, hidden_dim)
        
        if use_context:
            # Contextual relationship modeling
            self.context_attention = RelationAttention(hidden_dim)
            self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Relationship classification
        self.relationship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_predicates)
        )
    
    def forward(
        self,
        object_features: torch.Tensor,
        subject_indices: torch.Tensor,
        object_indices: torch.Tensor,
        context_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            object_features: Object features [batch_size, num_objects, feature_dim].
            subject_indices: Subject object indices [batch_size, num_relations].
            object_indices: Object indices [batch_size, num_relations].
            context_features: Optional context features.
            
        Returns:
            Relationship logits [batch_size, num_relations, num_predicates].
        """
        batch_size, num_relations = subject_indices.size(0), subject_indices.size(1)
        
        # Extract subject and object features
        subject_features = []
        object_features_list = []
        
        for b in range(batch_size):
            batch_subject_feats = []
            batch_object_feats = []
            
            for r in range(num_relations):
                subj_idx = subject_indices[b, r]
                obj_idx = object_indices[b, r]
                
                batch_subject_feats.append(object_features[b, subj_idx])
                batch_object_feats.append(object_features[b, obj_idx])
            
            subject_features.append(torch.stack(batch_subject_feats))
            object_features_list.append(torch.stack(batch_object_feats))
        
        subject_features = torch.stack(subject_features)  # [B, R, D]
        object_features_list = torch.stack(object_features_list)  # [B, R, D]
        
        # Project features
        subject_proj = self.subject_proj(subject_features)
        object_proj = self.object_proj(object_features_list)
        
        if self.use_context and context_features is not None:
            # Apply contextual attention
            context_subj = self.context_attention(subject_proj, context_features, context_features)
            context_obj = self.context_attention(object_proj, context_features, context_features)
            
            # Combine with context
            subject_proj = self.context_proj(torch.cat([subject_proj, context_subj], dim=-1))
            object_proj = self.context_proj(torch.cat([object_proj, context_obj], dim=-1))
        
        # Combine subject and object features
        combined_features = torch.cat([subject_proj, object_proj], dim=-1)
        
        # Predict relationships
        relationship_logits = self.relationship_classifier(combined_features)
        
        return relationship_logits
