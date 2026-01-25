"""Utility functions for logging and visualization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from PIL import Image


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level.
        log_file: Optional log file path.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("scene_graph")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_scene_graph_visualization(
    image: Union[np.ndarray, Image.Image, torch.Tensor],
    objects: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    output_path: str,
    show_scores: bool = True,
    figsize: tuple = (12, 8)
) -> None:
    """Save a visualization of the scene graph.
    
    Args:
        image: Input image.
        objects: List of detected objects with bboxes and labels.
        relationships: List of relationships between objects.
        output_path: Path to save the visualization.
        show_scores: Whether to show confidence scores.
        figsize: Figure size for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Plot image with bounding boxes
    ax1.imshow(image)
    ax1.set_title("Detected Objects")
    ax1.axis("off")
    
    # Draw bounding boxes
    for obj in objects:
        bbox = obj["bbox"]
        label = obj["label"]
        score = obj.get("score", 1.0)
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = plt.Rectangle(
            (x1, y1), width, height,
            fill=False, edgecolor="red", linewidth=2
        )
        ax1.add_patch(rect)
        
        text = label
        if show_scores:
            text += f" ({score:.2f})"
        
        ax1.text(x1, y1 - 5, text, color="red", fontsize=8, weight="bold")
    
    # Create and plot scene graph
    G = nx.DiGraph()
    
    # Add nodes (objects)
    for i, obj in enumerate(objects):
        G.add_node(i, label=obj["label"], score=obj.get("score", 1.0))
    
    # Add edges (relationships)
    for rel in relationships:
        subj_idx = rel["subject_idx"]
        obj_idx = rel["object_idx"]
        predicate = rel["predicate"]
        score = rel.get("score", 1.0)
        
        G.add_edge(subj_idx, obj_idx, label=predicate, score=score)
    
    # Plot graph
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color="lightblue", 
                          node_size=1000, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color="gray", 
                           arrows=True, arrowsize=20, alpha=0.6)
    
    # Draw labels
    node_labels = {i: G.nodes[i]["label"] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax2, font_size=8)
    
    edge_labels = {(u, v): G.edges[u, v]["label"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax2, font_size=6)
    
    ax2.set_title("Scene Graph")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_results_json(
    results: Dict[str, Any],
    output_path: str
) -> None:
    """Save results to JSON file.
    
    Args:
        results: Results dictionary.
        output_path: Path to save the JSON file.
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)


def create_metrics_table(
    metrics: Dict[str, float],
    output_path: str
) -> None:
    """Create a metrics table visualization.
    
    Args:
        metrics: Dictionary of metric names and values.
        output_path: Path to save the table image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("tight")
    ax.axis("off")
    
    # Prepare data
    metric_names = list(metrics.keys())
    metric_values = [f"{v:.4f}" for v in metrics.values()]
    
    table_data = list(zip(metric_names, metric_values))
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(metric_names) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#40466e")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f1f1f2")
    
    plt.title("Evaluation Metrics", weight="bold", size=14, pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
