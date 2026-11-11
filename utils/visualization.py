"""
Azhar EyeSight - Visualization & Explainable AI (Grad-CAM) Module
Provides PyTorch Grad-CAM computation, heatmap overlays, confusion matrix plotting,
training curve generation, and interactive Plotly probability charts.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for server/background execution
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch ResNet models.
    
    Generates coarse visual localization heatmaps highlighting the salient retinal
    fundus regions that most strongly influenced the neural network's classification decision.
    
    Mathematical Formulation:
        Weights: alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
        Heatmap: L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch ResNet model (e.g., ResNet18).
            target_layer: The convolutional layer to inspect. Defaults to model.layer4[-1].
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # ResNet layer4 is the final convolutional block before global average pooling
            if hasattr(model, "layer4"):
                self.target_layer = model.layer4[-1]
            else:
                raise ValueError("Target layer could not be determined automatically for this architecture.")
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple of gradients w.r.t layer outputs
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a given input tensor and target class.
        
        Args:
            input_tensor: Batch tensor of shape [1, 3, 224, 224].
            target_class_idx: Index of class to explain. If None, uses model's predicted class.
            
        Returns:
            np.ndarray: 2D float array in range [0.0, 1.0] of shape [224, 224].
        """
        # Ensure gradient tracking is enabled for input and model head
        self.model.zero_grad()
        
        # Clone tensor with requires_grad=True
        tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured by hooks.")

        # Global average pooling of gradients: [1, C, H, W] -> [C]
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = torch.mean(gradients, dim=(1, 2))  # [C]

        # Linear combination of activation channels weighted by pooled gradients
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain features that positively correlate with the target class
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Upscale heatmap to match input dimension (224x224)
        heatmap = cv2.resize(cam_np, config.IMAGE_SIZE)
        return heatmap

    def remove_hooks(self) -> None:
        """Unregisters PyTorch hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()


def overlay_gradcam_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Overlays a normalized Grad-CAM heatmap onto a PIL retinal fundus image.
    
    Args:
        pil_image: Original RGB PIL image.
        heatmap: 2D numpy array [0.0, 1.0] of shape matching pil_image or (224, 224).
        alpha: Heatmap blend weight (0.0 to 1.0).
        colormap: OpenCV colormap constant (default: COLORMAP_JET).
        
    Returns:
        PIL.Image: Combined blended RGB image.
    """
    # Resize original image to match heatmap or vice versa
    img_resized = pil_image.resize(config.IMAGE_SIZE)
    img_np = np.array(img_resized)

    # Scale heatmap to [0, 255] uint8
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha blend
    blended = np.uint8(alpha * heatmap_colored + (1.0 - alpha) * img_np)
    return Image.fromarray(blended)


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str] = config.CLASSES,
    save_path: Optional[Union[str, Path]] = config.CONFUSION_MATRIX_PATH
) -> plt.Figure:
    """
    Generates a healthcare-styled confusion matrix heatmap.
    """
    cm_np = np.array(cm)
    display_names = [config.CLASS_LABELS.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sns.heatmap(
        cm_np,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar=True,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Azhar EyeSight – Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis Category", fontsize=11, labelpad=8)
    ax.set_ylabel("True Verified Category", fontsize=11, labelpad=8)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, bbox_inches="tight")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = config.TRAINING_HISTORY_PATH
) -> plt.Figure:
    """
    Plots dual training curves: Loss (train vs val) and Accuracy (train vs val).
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Loss plot
    ax1.plot(epochs, history.get("train_loss", []), "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history.get("val_loss", []), "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy plot
    ax2.plot(epochs, history.get("train_acc", []), "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history.get("val_acc", []), "s--", color="#d62728", label="Val Accuracy", linewidth=2)
    ax2.set_title("Classification Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.suptitle("Azhar EyeSight – ResNet18 Training & Validation Progression", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, bbox_inches="tight")

    return fig


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    class_names: List[str] = config.CLASSES,
    save_path: Optional[Union[str, Path]] = config.PER_CLASS_METRICS_PATH
) -> plt.Figure:
    """
    Generates a grouped bar chart comparing Precision, Recall, and F1-score per disease category.
    """
    categories = [c for c in class_names if c in report_dict]
    display_names = [config.CLASS_LABELS.get(c, c) for c in categories]

    precisions = [report_dict[c]["precision"] * 100 for c in categories]
    recalls = [report_dict[c]["recall"] * 100 for c in categories]
    f1s = [report_dict[c]["f1-score"] * 100 for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width, precisions, width, label="Precision (%)", color="#0077b6")
    ax.bar(x, recalls, width, label="Recall (%)", color="#00b4d8")
    ax.bar(x + width, f1s, width, label="F1-Score (%)", color="#90e0ef")

    ax.set_title("Azhar EyeSight – Per-Class Diagnostic Performance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(target, bbox_inches="tight")

    return fig


def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """
    Creates an interactive, healthcare-styled horizontal bar chart of predicted class probabilities.
    
    Args:
        probabilities: Dict mapping class_name -> probability float in [0.0, 1.0].
        
    Returns:
        Plotly Figure.
    """
    labels = [config.CLASS_LABELS.get(k, k) for k in probabilities.keys()]
    values = [v * 100 for v in probabilities.values()]

    # Healthcare color palette: normal gets calm cyan/green; pathologies get clinical accents
    colors = []
    for k in probabilities.keys():
        if k == "normal":
            colors.append("#2a9d8f")  # Teal / green
        elif k == "cataract":
            colors.append("#e76f51")  # Coral
        elif k == "diabetic_retinopathy":
            colors.append("#d62828")  # Ruby red
        elif k == "glaucoma":
            colors.append("#f4a261")  # Amber
        else:
            colors.append("#457b9d")

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#1d3557", width=1.2)
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Confidence Probability: %{x:.2f}%<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(
            text="Model Class Probability Distribution",
            font=dict(size=15, color="#1d3557", family="Segoe UI, sans-serif")
        ),
        xaxis=dict(
            title="Calculated Softmax Probability (%)",
            range=[0, 110],
            showgrid=True,
            gridcolor="#e9ecef"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, family="Segoe UI, sans-serif")
        ),
        margin=dict(l=20, r=40, t=50, b=30),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=260
    )

    return fig
