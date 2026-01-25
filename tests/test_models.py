"""Tests for scene graph generation models."""

import pytest
import torch
import numpy as np

from src.models.scene_graph import MotifNet
from src.models.layers import GraphConvolution, ObjectDetector, RelationshipHead
from src.data.structures import SceneGraphBatch, SceneGraphMetrics
from src.utils.device import get_device, set_seed


class TestMotifNet:
    """Test cases for MotifNet model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device("cpu")
        
        self.model = MotifNet(
            backbone="resnet50",
            num_object_classes=10,
            num_predicate_classes=5,
            hidden_dim=64,
            num_layers=2,
            max_objects=10,
            max_relationships=20
        ).to(self.device)
        
        self.batch_size = 2
        self.image_size = (3, 224, 224)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.num_object_classes == 10
        assert self.model.num_predicate_classes == 5
        assert self.model.hidden_dim == 64
    
    def test_forward_pass(self):
        """Test forward pass."""
        images = torch.randn(self.batch_size, *self.image_size).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
        
        assert "object_logits" in outputs
        assert "relationship_logits" in outputs
        assert "object_features" in outputs
        assert "valid_objects" in outputs
        
        # Check output shapes
        assert outputs["object_logits"].shape == (self.batch_size, 10, 10)  # batch, max_objects, num_classes
        assert outputs["relationship_logits"].shape == (self.batch_size, 20, 5)  # batch, max_relationships, num_predicates
        assert outputs["object_features"].shape == (self.batch_size, 10, 64)  # batch, max_objects, hidden_dim
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        images = torch.randn(self.batch_size, *self.image_size).to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(images)
        
        assert features.shape[0] == self.batch_size
        assert features.shape[1] == 2048  # ResNet50 feature dimension
        assert len(features.shape) == 4  # B, C, H, W
    
    def test_object_detection(self):
        """Test object detection."""
        images = torch.randn(self.batch_size, *self.image_size).to(self.device)
        
        with torch.no_grad():
            features = self.model.extract_features(images)
            object_logits, bbox_deltas, object_features = self.model.detect_objects(features)
        
        assert object_logits.shape == (self.batch_size, 10, 10)
        assert bbox_deltas.shape == (self.batch_size, 10, 4)
        assert object_features.shape == (self.batch_size, 10, 64)
    
    def test_relationship_prediction(self):
        """Test relationship prediction."""
        batch_size = 2
        num_objects = 5
        hidden_dim = 64
        
        object_features = torch.randn(batch_size, num_objects, hidden_dim).to(self.device)
        object_logits = torch.randn(batch_size, num_objects, 10).to(self.device)
        valid_objects = torch.ones(batch_size, num_objects, dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            relationship_logits = self.model.predict_relationships(
                object_features, object_logits, valid_objects
            )
        
        assert relationship_logits.shape[0] == batch_size
        assert relationship_logits.shape[2] == 5  # num_predicate_classes


class TestGraphConvolution:
    """Test cases for GraphConvolution layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.layer = GraphConvolution(64, 64)
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        num_nodes = 5
        in_features = 64
        
        x = torch.randn(batch_size, num_nodes, in_features)
        adj = torch.ones(batch_size, num_nodes, num_nodes) / num_nodes  # Normalized adjacency
        
        output = self.layer(x, adj)
        
        assert output.shape == (batch_size, num_nodes, 64)
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        assert self.layer.weight.shape == (64, 64)
        assert self.layer.bias.shape == (64,)


class TestObjectDetector:
    """Test cases for ObjectDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ObjectDetector(
            in_channels=256,
            num_classes=10,
            roi_pool_size=7,
            hidden_dim=64
        )
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        num_rois = 5
        
        features = torch.randn(batch_size, 256, 32, 32)
        rois = torch.tensor([
            [[0, 0, 16, 16], [8, 8, 24, 24], [16, 16, 32, 32], [0, 0, 16, 16], [8, 8, 24, 24]],
            [[0, 0, 16, 16], [8, 8, 24, 24], [16, 16, 32, 32], [0, 0, 16, 16], [8, 8, 24, 24]]
        ], dtype=torch.float32)
        
        class_logits, bbox_deltas = self.detector(features, rois)
        
        assert class_logits.shape == (batch_size, num_rois, 10)
        assert bbox_deltas.shape == (batch_size, num_rois, 4)


class TestRelationshipHead:
    """Test cases for RelationshipHead."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.head = RelationshipHead(
            object_feature_dim=64,
            num_predicates=5,
            hidden_dim=64,
            use_context=True
        )
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        num_objects = 5
        num_relations = 10
        
        object_features = torch.randn(batch_size, num_objects, 64)
        subject_indices = torch.randint(0, num_objects, (batch_size, num_relations))
        object_indices = torch.randint(0, num_objects, (batch_size, num_relations))
        
        relationship_logits = self.head(object_features, subject_indices, object_indices)
        
        assert relationship_logits.shape == (batch_size, num_relations, 5)


class TestSceneGraphBatch:
    """Test cases for SceneGraphBatch data structure."""
    
    def test_batch_creation(self):
        """Test batch creation."""
        batch_size = 2
        num_objects = 5
        num_relationships = 10
        
        batch = SceneGraphBatch(
            images=torch.randn(batch_size, 3, 224, 224),
            object_boxes=torch.randn(batch_size, num_objects, 4),
            object_labels=torch.randint(0, 10, (batch_size, num_objects)),
            object_scores=torch.rand(batch_size, num_objects),
            relationship_triplets=torch.randint(0, num_objects, (batch_size, num_relationships, 3)),
            relationship_scores=torch.rand(batch_size, num_relationships),
            valid_objects=torch.ones(batch_size, num_objects, dtype=torch.bool),
            valid_relationships=torch.ones(batch_size, num_relationships, dtype=torch.bool)
        )
        
        assert batch.images.shape == (batch_size, 3, 224, 224)
        assert batch.object_boxes.shape == (batch_size, num_objects, 4)
        assert batch.object_labels.shape == (batch_size, num_objects)
        assert batch.relationship_triplets.shape == (batch_size, num_relationships, 3)


class TestSceneGraphMetrics:
    """Test cases for SceneGraphMetrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = SceneGraphMetrics(
            num_object_classes=10,
            num_predicate_classes=5
        )
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        assert self.metrics.num_object_classes == 10
        assert self.metrics.num_predicate_classes == 5
        assert self.metrics.object_correct == 0
        assert self.metrics.object_total == 0
    
    def test_metrics_update(self):
        """Test metrics update."""
        batch_size = 2
        num_objects = 5
        num_relationships = 10
        
        pred_objects = torch.randint(0, 10, (batch_size, num_objects))
        pred_relationships = torch.randint(0, 5, (batch_size, num_relationships))
        target_objects = torch.randint(0, 10, (batch_size, num_objects))
        target_relationships = torch.randint(0, 5, (batch_size, num_relationships))
        valid_objects = torch.ones(batch_size, num_objects, dtype=torch.bool)
        valid_relationships = torch.ones(batch_size, num_relationships, dtype=torch.bool)
        
        self.metrics.update(
            pred_objects, pred_relationships,
            target_objects, target_relationships,
            valid_objects, valid_relationships
        )
        
        assert self.metrics.object_total > 0
        assert self.metrics.relationship_total > 0
    
    def test_metrics_compute(self):
        """Test metrics computation."""
        # Update with some dummy data
        batch_size = 2
        num_objects = 5
        num_relationships = 10
        
        pred_objects = torch.randint(0, 10, (batch_size, num_objects))
        pred_relationships = torch.randint(0, 5, (batch_size, num_relationships))
        target_objects = torch.randint(0, 10, (batch_size, num_objects))
        target_relationships = torch.randint(0, 5, (batch_size, num_relationships))
        valid_objects = torch.ones(batch_size, num_objects, dtype=torch.bool)
        valid_relationships = torch.ones(batch_size, num_relationships, dtype=torch.bool)
        
        self.metrics.update(
            pred_objects, pred_relationships,
            target_objects, target_relationships,
            valid_objects, valid_relationships
        )
        
        results = self.metrics.compute()
        
        assert "object_accuracy" in results
        assert "relationship_accuracy" in results
        assert "mean_object_class_accuracy" in results
        assert "mean_predicate_class_accuracy" in results
        
        # Check that all values are between 0 and 1
        for key, value in results.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} is not between 0 and 1"


@pytest.mark.slow
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction."""
        set_seed(42)
        device = get_device("cpu")
        
        model = MotifNet(
            backbone="resnet50",
            num_object_classes=10,
            num_predicate_classes=5,
            hidden_dim=64,
            max_objects=5,
            max_relationships=10
        ).to(device)
        
        model.eval()
        
        # Create dummy input
        images = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Verify outputs
        assert "object_logits" in outputs
        assert "relationship_logits" in outputs
        assert outputs["object_logits"].shape[0] == 1
        assert outputs["relationship_logits"].shape[0] == 1
    
    def test_training_step(self):
        """Test a single training step."""
        set_seed(42)
        device = get_device("cpu")
        
        model = MotifNet(
            backbone="resnet50",
            num_object_classes=10,
            num_predicate_classes=5,
            hidden_dim=64,
            max_objects=5,
            max_relationships=10
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy batch
        batch = SceneGraphBatch(
            images=torch.randn(2, 3, 224, 224).to(device),
            object_boxes=torch.randn(2, 5, 4).to(device),
            object_labels=torch.randint(0, 10, (2, 5)).to(device),
            object_scores=torch.rand(2, 5).to(device),
            relationship_triplets=torch.randint(0, 5, (2, 10, 3)).to(device),
            relationship_scores=torch.rand(2, 10).to(device),
            valid_objects=torch.ones(2, 5, dtype=torch.bool).to(device),
            valid_relationships=torch.ones(2, 10, dtype=torch.bool).to(device)
        )
        
        # Forward pass
        outputs = model(batch.images)
        
        # Compute loss (simplified)
        object_loss = torch.nn.functional.cross_entropy(
            outputs["object_logits"].view(-1, 10),
            batch.object_labels.view(-1),
            ignore_index=-1
        )
        
        relationship_loss = torch.nn.functional.cross_entropy(
            outputs["relationship_logits"].view(-1, 5),
            batch.relationship_triplets[:, :, 2].view(-1),
            ignore_index=-1
        )
        
        total_loss = object_loss + relationship_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Verify loss is finite
        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0
