"""
Azhar EyeSight - PyTorch ResNet18 Model Architecture and Transfer Learning Utilities
Implements model creation, layer freezing/unfreezing, checkpoint saving, and robust loading.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torchvision import models

import config


def get_device() -> torch.device:
    """
    Automatically detects and returns available acceleration device (CUDA GPU or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_features: bool = True
) -> nn.Module:
    """
    Constructs a ResNet18 deep convolutional neural network for retinal image classification.
    
    Architecture & Transfer Learning Strategy:
    - Base: ResNet18 feature extractor pretrained on ImageNet (1000 categories).
    - Features: 18 layers with residual skip connections that mitigate vanishing gradients.
    - Transfer Learning: Freezes early convolutional blocks (layers 1-4) by default so that
      generic visual features (edges, textures, contours) are preserved.
    - Classification Head: Replaces final FC layer with a Linear layer (512 -> num_classes).
    
    Args:
        num_classes: Number of target retinal disease classes (default: 4).
        pretrained: If True, initializes weights from ImageNet pretraining.
        freeze_features: If True, freezes all backbone convolutional layers, training only the head.
        
    Returns:
        PyTorch nn.Module instance.
    """
    # Use modern torchvision weights enum if available, with backwards-compatible fallback
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=pretrained)

    # Freeze feature extraction layers if requested
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Ensure classification head parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_model_layers(model: nn.Module, unfreeze_layer4_only: bool = True) -> None:
    """
    Configurable fine-tuning helper: unfreezes deeper convolutional layers
    to adapt domain-specific retinal fundus feature representations.
    
    Args:
        model: ResNet18 model instance.
        unfreeze_layer4_only: If True, only unfreezes the deepest residual block (layer4).
                              If False, unfreezes all backbone layers.
    """
    if unfreeze_layer4_only:
        # Keep layers 1-3 frozen, unfreeze layer 4 and fc
        if hasattr(model, "layer4"):
            for param in model.layer4.parameters():
                param.requires_grad = True
    else:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

    # Always ensure classification head is trainable
    for param in model.fc.parameters():
        param.requires_grad = True


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    val_acc: float,
    class_to_idx: Dict[str, int],
    path: Union[str, Path] = config.MODEL_PATH,
    extra_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Persists model weights and complete training metadata into a checkpoint file (.pth).
    
    Saved attributes:
    - model_state_dict: Network weights
    - optimizer_state_dict: Optimizer state (if available)
    - epoch: Training epoch reached
    - val_loss: Validation loss at checkpoint
    - val_acc: Validation accuracy at checkpoint
    - class_to_idx: Label mapping dictionary
    - classes: List of class names
    - config: Training hyperparameters
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_architecture": "resnet18",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "class_to_idx": class_to_idx,
        "classes": list(class_to_idx.keys()),
        "num_classes": len(class_to_idx),
        "config": extra_config or {}
    }

    torch.save(checkpoint, target_path)


def load_checkpoint(
    model_path: Union[str, Path] = config.MODEL_PATH,
    device: Optional[torch.device] = None,
    num_classes: int = config.NUM_CLASSES
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Safely loads a trained ResNet18 model and associated metadata from disk.
    
    Handles:
    - Device mapping (CUDA to CPU or vice versa)
    - Both full checkpoint dictionaries and raw state_dicts
    - Missing or corrupted file errors
    
    Returns:
        tuple: (model, checkpoint_info_dict)
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint cannot be loaded.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights file not found at '{path}'. "
            "Please train the model first with 'python train.py' or provide a valid checkpoint."
        )

    if device is None:
        device = get_device()

    try:
        checkpoint = torch.save if False else torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load PyTorch checkpoint from '{path}': {e}")

    # Inspect checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full metadata checkpoint
        saved_num_classes = checkpoint.get("num_classes", num_classes)
        model = build_model(num_classes=saved_num_classes, pretrained=False, freeze_features=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        info = checkpoint
    elif isinstance(checkpoint, dict):
        # Raw state dict
        model = build_model(num_classes=num_classes, pretrained=False, freeze_features=False)
        model.load_state_dict(checkpoint)
        info = {
            "classes": config.CLASSES,
            "class_to_idx": {c: i for i, c in enumerate(config.CLASSES)},
            "epoch": None,
            "val_loss": None,
            "val_acc": None
        }
    else:
        # Serialized entire nn.Module
        model = checkpoint
        info = {
            "classes": config.CLASSES,
            "class_to_idx": {c: i for i, c in enumerate(config.CLASSES)}
        }

    model.to(device)
    model.eval()
    return model, info
