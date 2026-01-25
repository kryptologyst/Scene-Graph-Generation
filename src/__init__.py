"""Scene Graph Generation Package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.scene_graph import MotifNet
from .data.datasets import VisualGenomeDataset
from .train.trainer import SceneGraphTrainer
from .eval.evaluator import SceneGraphEvaluator

__all__ = [
    "MotifNet",
    "VisualGenomeDataset", 
    "SceneGraphTrainer",
    "SceneGraphEvaluator",
]
