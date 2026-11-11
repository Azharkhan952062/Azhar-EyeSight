"""
Azhar EyeSight Utilities Package
Exports preprocessing, model architecture, visualization, and metric calculation modules.
"""

from .preprocessing import (
    get_train_transforms,
    get_val_test_transforms,
    preprocess_image_for_inference,
    validate_dataset_structure,
    safe_load_image
)

from .model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    unfreeze_model_layers,
    get_device
)

from .visualization import (
    GradCAM,
    overlay_gradcam_on_image,
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_metrics,
    create_probability_chart
)

from .metrics import (
    calculate_metrics,
    save_metrics_json,
    load_metrics_json
)

__all__ = [
    "get_train_transforms",
    "get_val_test_transforms",
    "preprocess_image_for_inference",
    "validate_dataset_structure",
    "safe_load_image",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "unfreeze_model_layers",
    "get_device",
    "GradCAM",
    "overlay_gradcam_on_image",
    "plot_confusion_matrix",
    "plot_training_history",
    "plot_per_class_metrics",
    "create_probability_chart",
    "calculate_metrics",
    "save_metrics_json",
    "load_metrics_json",
]
