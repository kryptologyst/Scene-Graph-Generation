#!/usr/bin/env python3
"""Evaluation script for scene graph generation."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from src.data.datasets import VisualGenomeDataset, collate_fn
from src.models.scene_graph import MotifNet
from src.eval.evaluator import SceneGraphEvaluator
from src.utils.device import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate scene graph generation model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4,
        help="Number of data loader workers"
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
                "image_size": [512, 512],
                "min_image_size": 224,
                "max_image_size": 1024,
                "max_objects_per_image": 50,
                "max_relationships_per_image": 100,
                "min_object_area": 0.001,
                "min_relationship_area": 0.0001
            }
        }
    
    return OmegaConf.load(config_path)


def create_test_loader(config: Dict[str, Any], batch_size: int, num_workers: int) -> DataLoader:
    """Create test data loader."""
    data_config = config["data"]
    
    # Create test dataset
    test_dataset = VisualGenomeDataset(
        data_dir=data_config["data_dir"],
        image_dir=data_config["image_dir"],
        annotation_file=data_config["annotation_file"],
        image_size=tuple(data_config["image_size"]),
        min_image_size=data_config["min_image_size"],
        max_image_size=data_config["max_image_size"],
        use_augmentation=False,  # No augmentation for testing
        split="test"
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return test_loader


def load_model(checkpoint_path: str, config: Dict[str, Any]) -> MotifNet:
    """Load trained model from checkpoint."""
    model_config = config["model"]
    
    # Create model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader = create_test_loader(config, args.batch_size, args.num_workers)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config)
    
    # Print model info
    from src.utils.device import count_parameters, get_model_size_mb
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Get class names
    object_names, predicate_names = test_loader.dataset.get_class_names()
    print(f"Object classes: {len(object_names)}")
    print(f"Predicate classes: {len(predicate_names)}")
    
    # Create evaluator
    evaluator = SceneGraphEvaluator(
        model=model,
        test_loader=test_loader,
        device=args.device,
        output_dir=args.output_dir,
        class_names=(object_names, predicate_names)
    )
    
    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate()
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
