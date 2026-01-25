"""Dataset classes for scene graph generation."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..data.structures import (
    ObjectAnnotation,
    RelationshipAnnotation,
    SceneGraphAnnotation,
    SceneGraphBatch
)


class VisualGenomeDataset(Dataset):
    """Visual Genome dataset for scene graph generation.
    
    This dataset loads images and their scene graph annotations from the Visual Genome dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_dir: str,
        annotation_file: str,
        image_size: Tuple[int, int] = (512, 512),
        min_image_size: int = 224,
        max_image_size: int = 1024,
        use_augmentation: bool = True,
        horizontal_flip: float = 0.5,
        color_jitter: float = 0.1,
        rotation: int = 10,
        scale: Tuple[float, float] = (0.8, 1.2),
        max_objects_per_image: int = 50,
        max_relationships_per_image: int = 100,
        min_object_area: float = 0.001,
        min_relationship_area: float = 0.0001,
        split: str = "train"
    ):
        """Initialize Visual Genome dataset.
        
        Args:
            data_dir: Root data directory.
            image_dir: Directory containing images.
            annotation_file: Path to annotation JSON file.
            image_size: Target image size (height, width).
            min_image_size: Minimum image size.
            max_image_size: Maximum image size.
            use_augmentation: Whether to use data augmentation.
            horizontal_flip: Probability of horizontal flip.
            color_jitter: Color jitter strength.
            rotation: Maximum rotation angle in degrees.
            scale: Scale range for augmentation.
            max_objects_per_image: Maximum objects per image.
            max_relationships_per_image: Maximum relationships per image.
            min_object_area: Minimum object area ratio.
            min_relationship_area: Minimum relationship area ratio.
            split: Dataset split (train/val/test).
        """
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.max_objects_per_image = max_objects_per_image
        self.max_relationships_per_image = max_relationships_per_image
        self.min_object_area = min_object_area
        self.min_relationship_area = min_relationship_area
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Create class mappings
        self.object_classes, self.predicate_classes = self._create_class_mappings()
        
        # Setup transforms
        self.transform = self._setup_transforms(
            use_augmentation, horizontal_flip, color_jitter, rotation, scale
        )
    
    def _load_annotations(self) -> List[SceneGraphAnnotation]:
        """Load annotations from JSON file."""
        if not self.annotation_file.exists():
            # Create dummy annotations for demo
            return self._create_dummy_annotations()
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for item in data:
            # Extract objects
            objects = []
            for obj in item.get('objects', []):
                bbox = obj['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                objects.append(ObjectAnnotation(
                    bbox=bbox,
                    label=obj['name'],
                    score=obj.get('score', 1.0),
                    attributes=obj.get('attributes', [])
                ))
            
            # Extract relationships
            relationships = []
            for rel in item.get('relationships', []):
                relationships.append(RelationshipAnnotation(
                    subject_idx=rel['subject'],
                    object_idx=rel['object'],
                    predicate=rel['predicate'],
                    score=rel.get('score', 1.0)
                ))
            
            annotations.append(SceneGraphAnnotation(
                image_id=item['image_id'],
                image_path=str(self.image_dir / f"{item['image_id']}.jpg"),
                objects=objects,
                relationships=relationships,
                image_size=(item['width'], item['height'])
            ))
        
        return annotations
    
    def _create_dummy_annotations(self) -> List[SceneGraphAnnotation]:
        """Create dummy annotations for demo purposes."""
        annotations = []
        
        # Create some dummy images and annotations
        for i in range(10):
            # Create dummy objects
            objects = [
                ObjectAnnotation(
                    bbox=[50, 50, 150, 150],
                    label="person",
                    score=0.9,
                    attributes=["standing"]
                ),
                ObjectAnnotation(
                    bbox=[200, 100, 300, 200],
                    label="car",
                    score=0.8,
                    attributes=["red"]
                )
            ]
            
            # Create dummy relationships
            relationships = [
                RelationshipAnnotation(
                    subject_idx=0,
                    object_idx=1,
                    predicate="near",
                    score=0.7
                )
            ]
            
            annotations.append(SceneGraphAnnotation(
                image_id=f"dummy_{i}",
                image_path=f"dummy_{i}.jpg",
                objects=objects,
                relationships=relationships,
                image_size=(512, 512)
            ))
        
        return annotations
    
    def _create_class_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Create class name to index mappings."""
        # Collect all unique classes
        object_classes = set()
        predicate_classes = set()
        
        for ann in self.annotations:
            for obj in ann.objects:
                object_classes.add(obj.label)
            for rel in ann.relationships:
                predicate_classes.add(rel.predicate)
        
        # Create mappings
        object_class_to_idx = {cls: idx for idx, cls in enumerate(sorted(object_classes))}
        predicate_class_to_idx = {cls: idx for idx, cls in enumerate(sorted(predicate_classes))}
        
        return object_class_to_idx, predicate_class_to_idx
    
    def _setup_transforms(
        self,
        use_augmentation: bool,
        horizontal_flip: float,
        color_jitter: float,
        rotation: int,
        scale: Tuple[float, float]
    ) -> transforms.Compose:
        """Setup image transforms."""
        transform_list = []
        
        if use_augmentation and self.split == "train":
            # Random horizontal flip
            if horizontal_flip > 0:
                transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
            
            # Color jitter
            if color_jitter > 0:
                transform_list.append(transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter
                ))
            
            # Random rotation
            if rotation > 0:
                transform_list.append(transforms.RandomRotation(rotation))
            
            # Random scale and crop
            transform_list.append(transforms.RandomResizedCrop(
                self.image_size,
                scale=scale,
                ratio=(0.8, 1.2)
            ))
        else:
            # Resize for validation/test
            transform_list.append(transforms.Resize(self.image_size))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        annotation = self.annotations[idx]
        
        # Load image
        try:
            image = Image.open(annotation.image_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # Create dummy image if file doesn't exist
            image = Image.new('RGB', (512, 512), color='white')
        
        # Apply transforms
        image = self.transform(image)
        
        # Process objects
        objects = annotation.objects[:self.max_objects_per_image]
        object_boxes = []
        object_labels = []
        object_scores = []
        
        for obj in objects:
            # Normalize bbox coordinates
            x1, y1, x2, y2 = obj.bbox
            width, height = annotation.image_size
            
            # Normalize to [0, 1]
            x1_norm = x1 / width
            y1_norm = y1 / height
            x2_norm = x2 / width
            y2_norm = y2 / height
            
            # Check minimum area
            area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
            if area >= self.min_object_area:
                object_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                object_labels.append(self.object_classes.get(obj.label, 0))
                object_scores.append(obj.score)
        
        # Pad objects
        while len(object_boxes) < self.max_objects_per_image:
            object_boxes.append([0, 0, 0, 0])
            object_labels.append(0)
            object_scores.append(0.0)
        
        # Process relationships
        relationships = annotation.relationships[:self.max_relationships_per_image]
        relationship_triplets = []
        relationship_scores = []
        
        for rel in relationships:
            if (rel.subject_idx < len(objects) and 
                rel.object_idx < len(objects) and
                rel.subject_idx != rel.object_idx):
                
                relationship_triplets.append([
                    rel.subject_idx,
                    rel.object_idx,
                    self.predicate_classes.get(rel.predicate, 0)
                ])
                relationship_scores.append(rel.score)
        
        # Pad relationships
        while len(relationship_triplets) < self.max_relationships_per_image:
            relationship_triplets.append([0, 0, 0])
            relationship_scores.append(0.0)
        
        # Create valid masks
        valid_objects = torch.tensor([
            i < len(objects) for i in range(self.max_objects_per_image)
        ], dtype=torch.bool)
        
        valid_relationships = torch.tensor([
            i < len(relationships) for i in range(self.max_relationships_per_image)
        ], dtype=torch.bool)
        
        return {
            "image": image,
            "object_boxes": torch.tensor(object_boxes, dtype=torch.float32),
            "object_labels": torch.tensor(object_labels, dtype=torch.long),
            "object_scores": torch.tensor(object_scores, dtype=torch.float32),
            "relationship_triplets": torch.tensor(relationship_triplets, dtype=torch.long),
            "relationship_scores": torch.tensor(relationship_scores, dtype=torch.float32),
            "valid_objects": valid_objects,
            "valid_relationships": valid_relationships,
            "image_id": annotation.image_id
        }
    
    def get_class_names(self) -> Tuple[List[str], List[str]]:
        """Get class names."""
        object_names = [cls for cls, _ in sorted(self.object_classes.items(), key=lambda x: x[1])]
        predicate_names = [cls for cls, _ in sorted(self.predicate_classes.items(), key=lambda x: x[1])]
        return object_names, predicate_names


def collate_fn(batch: List[Dict[str, Any]]) -> SceneGraphBatch:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples from dataset.
        
    Returns:
        Batched data.
    """
    images = torch.stack([item["image"] for item in batch])
    object_boxes = torch.stack([item["object_boxes"] for item in batch])
    object_labels = torch.stack([item["object_labels"] for item in batch])
    object_scores = torch.stack([item["object_scores"] for item in batch])
    relationship_triplets = torch.stack([item["relationship_triplets"] for item in batch])
    relationship_scores = torch.stack([item["relationship_scores"] for item in batch])
    valid_objects = torch.stack([item["valid_objects"] for item in batch])
    valid_relationships = torch.stack([item["valid_relationships"] for item in batch])
    
    return SceneGraphBatch(
        images=images,
        object_boxes=object_boxes,
        object_labels=object_labels,
        object_scores=object_scores,
        relationship_triplets=relationship_triplets,
        relationship_scores=relationship_scores,
        valid_objects=valid_objects,
        valid_relationships=valid_relationships
    )
