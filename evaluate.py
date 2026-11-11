"""
Azhar EyeSight - Model Evaluation Pipeline
Evaluates the trained ResNet18 model on the independent test dataset.
Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and generates visual plots.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from utils.preprocessing import get_val_test_transforms
from utils.model import load_checkpoint, get_device
from utils.metrics import calculate_metrics, save_metrics_json
from utils.visualization import plot_confusion_matrix, plot_per_class_metrics


def run_evaluation(
    model_path: Path = config.MODEL_PATH,
    test_dir: Path = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Runs full evaluation on the independent test dataset and saves all metrics & figures.
    """
    if device is None:
        device = get_device()

    print("\n" + "=" * 65)
    print(" AZHAR EYESIGHT – MODEL TEST EVALUATION PIPELINE")
    print("=" * 65)
    print(f"Device           : {device}")
    print(f"Model Checkpoint : {model_path.resolve()}")
    print(f"Test Directory   : {test_dir.resolve()}")
    print("=" * 65)

    # 1. Verify model checkpoint exists
    if not model_path.is_file():
        print(f"\n[ERROR] Model checkpoint not found at: {model_path.resolve()}")
        print("Evaluation cannot proceed without a trained model.")
        print("Please train the model first by running:\n  python train.py\n")
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # 2. Verify test dataset exists and has valid class folders
    if not test_dir.is_dir():
        print(f"\n[ERROR] Test directory not found at: {test_dir.resolve()}")
        print("Please place the test retinal images in data/test/ according to the dataset structure.")
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Inspect test classes
    test_transforms = get_val_test_transforms()
    try:
        test_dataset = ImageFolder(root=str(test_dir), transform=test_transforms)
    except Exception as e:
        print(f"\n[ERROR] Failed to load test dataset: {e}")
        raise

    if len(test_dataset) == 0:
        print(f"\n[ERROR] Test directory '{test_dir}' contains zero valid images.")
        print("Please place test images inside the respective class subdirectories:")
        for cls in config.CLASSES:
            print(f"  - data/test/{cls}/")
        raise ValueError(f"No test images found in {test_dir}")

    print(f"\n[Data] Total test samples : {len(test_dataset)}")
    print(f"[Data] Class mapping      : {test_dataset.class_to_idx}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # 3. Load Trained Model
    print("\nLoading trained model weights...")
    model, checkpoint_info = load_checkpoint(model_path=model_path, device=device)
    model.eval()

    # 4. Run Test Inference Loop
    y_true = []
    y_pred = []

    print("Running inference across test batches...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # 5. Compute Quantitative Metrics
    # Map dataset class indices to standardized class names
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    standardized_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=standardized_class_names
    )

    # Add metadata
    metrics["model_path"] = str(model_path.resolve())
    metrics["test_dir"] = str(test_dir.resolve())
    metrics["checkpoint_epoch"] = checkpoint_info.get("epoch")
    metrics["checkpoint_val_loss"] = checkpoint_info.get("val_loss")
    metrics["checkpoint_val_acc"] = checkpoint_info.get("val_acc")

    # 6. Display Metrics to Console
    print("\n" + "=" * 65)
    print(" EVALUATION RESULTS – CLASSIFICATION PERFORMANCE")
    print("=" * 65)
    print(f"Overall Accuracy       : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision        : {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall           : {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score         : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1-Score      : {metrics['weighted_f1'] * 100:.2f}%")
    print("-" * 65)
    print("Classification Report:")
    print(metrics["classification_report_str"])
    print("-" * 65)
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print("  ", row)
    print("=" * 65)

    # 7. Persist Evaluation Metrics & Plots
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, config.METRICS_JSON_PATH)
    print(f"\n[Saved] Metrics JSON saved to : {config.METRICS_JSON_PATH.resolve()}")

    try:
        plot_confusion_matrix(metrics["confusion_matrix"], class_names=standardized_class_names, save_path=config.CONFUSION_MATRIX_PATH)
        print(f"[Saved] Confusion Matrix plot : {config.CONFUSION_MATRIX_PATH.resolve()}")
    except Exception as e:
        print(f"[Warning] Failed to save confusion matrix plot: {e}")

    try:
        plot_per_class_metrics(metrics["classification_report_dict"], class_names=standardized_class_names, save_path=config.PER_CLASS_METRICS_PATH)
        print(f"[Saved] Per-Class chart       : {config.PER_CLASS_METRICS_PATH.resolve()}")
    except Exception as e:
        print(f"[Warning] Failed to save per-class metrics plot: {e}")

    print("\nEvaluation completed successfully!\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Azhar EyeSight – Model Evaluation Script")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH), help="Path to trained PyTorch weights")
    parser.add_argument("--test_dir", type=str, default=str(config.TEST_DIR), help="Path to test dataset directory")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")

    args = parser.parse_args()

    try:
        run_evaluation(
            model_path=Path(args.model),
            test_dir=Path(args.test_dir),
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"\n[Evaluation Terminated] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
