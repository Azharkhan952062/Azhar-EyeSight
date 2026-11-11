"""
Azhar EyeSight - Evaluation Metrics Module
Computes, serializes, and formats machine learning metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively converts NumPy types into native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types(x) for x in obj]
    return obj


def calculate_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: List[str] = config.CLASSES
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for retinal fundus classification.
    
    Metrics:
    - Overall Accuracy
    - Macro-averaged Precision, Recall, F1-score
    - Weighted-averaged Precision, Recall, F1-score
    - Per-class Precision, Recall, F1-score, and Support
    - Confusion Matrix (integer 2D list)
    - Full classification report dictionary and string
    """
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_true_np, y_pred_np))
    prec_macro = float(precision_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true_np, y_pred_np, labels=labels, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_np, y_pred_np, labels=labels, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_np, y_pred_np, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels).tolist()

    report_dict = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_str = classification_report(
        y_true_np,
        y_pred_np,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    # Per-class summary table
    per_class = {}
    for cls in class_names:
        if cls in report_dict:
            per_class[cls] = {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1_score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }

    metrics = {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm,
        "per_class": per_class,
        "classification_report_dict": _convert_numpy_types(report_dict),
        "classification_report_str": report_str,
        "class_names": class_names,
        "total_samples": int(len(y_true_np))
    }

    return metrics


def save_metrics_json(metrics: Dict[str, Any], path: Union[str, Path] = config.METRICS_JSON_PATH) -> None:
    """
    Saves evaluation metrics dictionary to a JSON file.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    clean_metrics = _convert_numpy_types(metrics)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=4)


def load_metrics_json(path: Union[str, Path] = config.METRICS_JSON_PATH) -> Optional[Dict[str, Any]]:
    """
    Loads saved evaluation metrics from a JSON file. Returns None if file does not exist.
    """
    target_path = Path(path)
    if not target_path.is_file():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
