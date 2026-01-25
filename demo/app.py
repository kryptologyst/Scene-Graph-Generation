"""Streamlit demo application for scene graph generation."""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from src.models.scene_graph import MotifNet
from src.utils.device import get_device
from src.utils.visualization import save_scene_graph_visualization


class SceneGraphDemo:
    """Demo application for scene graph generation."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize demo.
        
        Args:
            model_path: Path to trained model checkpoint.
        """
        self.model_path = model_path
        self.model = None
        self.device = get_device("auto")
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model.
        
        Args:
            model_path: Path to model checkpoint.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            self.model = MotifNet(
                num_object_classes=150,
                num_predicate_classes=50
            )
            
            # Load state dict
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            st.success(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            self.model = None
    
    def predict_scene_graph(self, image: Image.Image) -> Dict[str, Any]:
        """Predict scene graph for an image.
        
        Args:
            image: Input image.
            
        Returns:
            Dictionary containing predictions.
        """
        if self.model is None:
            # Return dummy predictions for demo
            return self._get_dummy_predictions()
        
        # Preprocess image
        transform = torch.nn.Sequential(
            torch.nn.Upsample(size=(512, 512)),
            torch.nn.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        
        # Convert PIL to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Apply transform
        image_tensor = transform(image_tensor)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        return self._process_predictions(predictions, image.size)
    
    def _get_dummy_predictions(self) -> Dict[str, Any]:
        """Get dummy predictions for demo."""
        return {
            "objects": [
                {
                    "bbox": [50, 50, 150, 150],
                    "label": "person",
                    "score": 0.9,
                    "attributes": ["standing"]
                },
                {
                    "bbox": [200, 100, 300, 200],
                    "label": "car",
                    "score": 0.8,
                    "attributes": ["red"]
                },
                {
                    "bbox": [100, 200, 200, 300],
                    "label": "tree",
                    "score": 0.7,
                    "attributes": ["green"]
                }
            ],
            "relationships": [
                {
                    "subject_idx": 0,
                    "object_idx": 1,
                    "predicate": "near",
                    "score": 0.7
                },
                {
                    "subject_idx": 0,
                    "object_idx": 2,
                    "predicate": "standing_by",
                    "score": 0.6
                }
            ]
        }
    
    def _process_predictions(
        self, 
        predictions: Dict[str, torch.Tensor], 
        image_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process model predictions.
        
        Args:
            predictions: Raw model predictions.
            image_size: Original image size (width, height).
            
        Returns:
            Processed predictions.
        """
        # Get object predictions
        object_logits = predictions["object_logits"][0]  # Remove batch dimension
        object_probs = torch.softmax(object_logits, dim=-1)
        object_preds = object_logits.argmax(dim=-1)
        
        # Get relationship predictions
        relationship_logits = predictions["relationship_logits"][0]
        relationship_probs = torch.softmax(relationship_logits, dim=-1)
        relationship_preds = relationship_logits.argmax(dim=-1)
        
        # Process objects
        objects = []
        valid_objects = predictions["valid_objects"][0]
        
        for i in range(len(valid_objects)):
            if valid_objects[i]:
                # Convert normalized coordinates back to image coordinates
                bbox = predictions["bbox_deltas"][0, i].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                # Scale to image size
                x1 *= image_size[0]
                y1 *= image_size[1]
                x2 *= image_size[0]
                y2 *= image_size[1]
                
                objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": f"object_{object_preds[i].item()}",
                    "score": object_probs[i].max().item(),
                    "attributes": []
                })
        
        # Process relationships
        relationships = []
        valid_relationships = predictions["valid_relationships"][0]
        
        for i in range(len(valid_relationships)):
            if valid_relationships[i]:
                # Get subject and object indices
                triplet = predictions["relationship_triplets"][0, i]
                subj_idx, obj_idx, pred_idx = triplet
                
                relationships.append({
                    "subject_idx": subj_idx.item(),
                    "object_idx": obj_idx.item(),
                    "predicate": f"predicate_{relationship_preds[i].item()}",
                    "score": relationship_probs[i].max().item()
                })
        
        return {
            "objects": objects,
            "relationships": relationships
        }


def create_scene_graph_plotly(objects: List[Dict], relationships: List[Dict]) -> go.Figure:
    """Create interactive scene graph visualization with Plotly.
    
    Args:
        objects: List of detected objects.
        relationships: List of relationships.
        
    Returns:
        Plotly figure.
    """
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, obj in enumerate(objects):
        G.add_node(i, label=obj["label"], score=obj["score"])
    
    # Add edges
    for rel in relationships:
        G.add_edge(
            rel["subject_idx"], 
            rel["object_idx"], 
            label=rel["predicate"], 
            score=rel["score"]
        )
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Extract edge positions
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get edge info
        edge_data = G.edges[edge]
        edge_info.append(f"{edge_data['label']} ({edge_data['score']:.2f})")
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="middle center",
        marker=dict(
            size=50,
            color='lightblue',
            line=dict(width=2, color='black')
        )
    )
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create edge label trace
    edge_label_trace = go.Scatter(
        x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
        y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
        mode='text',
        text=edge_info,
        textfont=dict(size=10),
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    
    fig.update_layout(
        title="Scene Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Interactive scene graph visualization",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Scene Graph Generation Demo",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Scene Graph Generation Demo")
    st.markdown("Upload an image to generate a scene graph showing objects and their relationships.")
    
    # Initialize demo
    if "demo" not in st.session_state:
        st.session_state.demo = SceneGraphDemo()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model upload
    uploaded_model = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=["pt", "pth"],
        help="Upload a trained model checkpoint (.pt or .pth file)"
    )
    
    if uploaded_model is not None:
        # Save uploaded model temporarily
        model_path = f"temp_model_{uploaded_model.name}"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        # Load model
        st.session_state.demo.load_model(model_path)
        
        # Clean up
        Path(model_path).unlink(missing_ok=True)
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to generate a scene graph"
    )
    
    if uploaded_image is not None:
        # Load image
        image = Image.open(uploaded_image).convert("RGB")
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
            
            # Generate scene graph
            if st.button("Generate Scene Graph", type="primary"):
                with st.spinner("Generating scene graph..."):
                    predictions = st.session_state.demo.predict_scene_graph(image)
                    st.session_state.predictions = predictions
        
        with col2:
            st.subheader("Scene Graph Visualization")
            
            if "predictions" in st.session_state:
                predictions = st.session_state.predictions
                
                # Create interactive plot
                fig = create_scene_graph_plotly(
                    predictions["objects"], 
                    predictions["relationships"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Objects", "Relationships", "Raw Data"])
                
                with tab1:
                    st.subheader("Detected Objects")
                    for i, obj in enumerate(predictions["objects"]):
                        with st.expander(f"Object {i+1}: {obj['label']} (Score: {obj['score']:.2f})"):
                            st.write(f"**Bounding Box:** {obj['bbox']}")
                            st.write(f"**Attributes:** {obj.get('attributes', [])}")
                
                with tab2:
                    st.subheader("Detected Relationships")
                    for i, rel in enumerate(predictions["relationships"]):
                        subj_obj = predictions["objects"][rel["subject_idx"]]["label"]
                        obj_obj = predictions["objects"][rel["object_idx"]]["label"]
                        st.write(f"**{subj_obj}** --{rel['predicate']}--> **{obj_obj}** (Score: {rel['score']:.2f})")
                
                with tab3:
                    st.subheader("Raw Predictions")
                    st.json(predictions)
                
                # Download results
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(predictions, indent=2),
                    file_name="scene_graph_results.json",
                    mime="application/json"
                )
    
    else:
        st.info("Please upload an image to get started.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "This demo showcases scene graph generation using deep learning models. "
        "Scene graphs represent images as structured graphs of objects and their relationships, "
        "enabling better understanding of visual scenes."
    )


if __name__ == "__main__":
    main()
