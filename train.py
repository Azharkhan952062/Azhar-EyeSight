"""
Azhar EyeSight - Model Training Pipeline
Implements Transfer Learning with ResNet18, Adam optimizer, CrossEntropyLoss,
learning rate scheduling, early stopping, and checkpoint management.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import config
from utils.preprocessing import (
    get_train_transforms,
    get_val_test_transforms,
    validate_dataset_structure
)
from utils.model import (
    build_model,
    unfreeze_model_layers,
    save_checkpoint,
    get_device
)
from utils.visualization import plot_training_history


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch across all training batches."""
    model.train()
    running_loss = 0.0
    correct_samples = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_samples += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = correct_samples / max(total_samples, 1)
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluates the model over the validation set without computing gradients."""
    model.eval()
    running_loss = 0.0
    correct_samples = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_samples += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_acc = correct_samples / max(total_samples, 1)
    return epoch_loss, epoch_acc


def run_training(
    data_dir: Path = config.DATA_DIR,
    model_save_path: Path = config.MODEL_PATH,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    epochs: int = config.NUM_EPOCHS,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    fine_tune: bool = False,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Executes the full ResNet18 training loop with early stopping.
    """
    if device is None:
        device = get_device()

    print("\n" + "=" * 65)
    print(" AZHAR EYESIGHT – RESNET18 MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"Device               : {device}")
    print(f"Dataset Location     : {data_dir.resolve()}")
    print(f"Model Checkpoint Path: {model_save_path.resolve()}")
    print(f"Batch Size           : {batch_size}")
    print(f"Learning Rate        : {learning_rate}")
    print(f"Epochs               : {epochs}")
    print(f"Early Stop Patience  : {patience}")
    print(f"Fine-tuning Mode     : {'Enabled (Layer4 unfreezed)' if fine_tune else 'Feature Extraction (Head only)'}")
    print("=" * 65)

    # 1. Dataset Verification
    validation_report = validate_dataset_structure(data_dir=data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not validation_report["is_valid"]:
        print("\n[ERROR] Dataset structure is incomplete or missing images.")
        print("Missing folders :", validation_report["missing_folders"])
        print("Empty classes   :", validation_report["empty_classes"])
        print("\nPlease follow the Dataset Setup Guide in README.md to place retinal fundus images.")
        raise FileNotFoundError(
            f"Dataset incomplete at {data_dir}. Required folders: train/ and val/ with classes {config.CLASSES}"
        )

    # 2. Datasets and DataLoaders
    train_transforms = get_train_transforms()
    val_transforms = get_val_test_transforms()

    train_dataset = ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_transforms)

    print(f"\n[Data] Training samples   : {len(train_dataset)}")
    print(f"[Data] Validation samples : {len(val_dataset)}")
    print(f"[Data] Class Mapping      : {train_dataset.class_to_idx}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # 3. Model Initialization
    model = build_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_features=True
    )

    if fine_tune:
        unfreeze_model_layers(model, unfreeze_layer4_only=True)

    model.to(device)

    # 4. Optimizer, Criterion, Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Model] Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Trainable Params: {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # 5. Training Loop with Early Stopping
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()
    print("\nStarting Training...\n" + "-" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"({epoch_duration:.1f}s)"
        )

        # Check for best model (using validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                class_to_idx=train_dataset.class_to_idx,
                path=model_save_path,
                extra_config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "fine_tune": fine_tune
                }
            )
            print(f"  --> Checkpoint saved: val_loss improved to {val_loss:.4f} (Val Acc: {val_acc * 100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Triggered after {epoch} epochs. Restoring best model from epoch {best_epoch}.")
            break

    total_training_time = time.time() - start_time
    print("-" * 65)
    print(f"Training completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Loss     : {best_val_loss:.4f}")
    print(f"Best Validation Accuracy : {best_val_acc * 100:.2f}% (Epoch {best_epoch})")

    # 6. Save Training History Plot and JSON
    try:
        plot_training_history(history, save_path=config.TRAINING_HISTORY_PATH)
        print(f"[Plot] Training history saved to: {config.TRAINING_HISTORY_PATH.resolve()}")
    except Exception as e:
        print(f"[Warning] Failed to generate training history plot: {e}")

    try:
        import json
        history_json_path = config.RESULTS_DIR / "training_history.json"
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"[JSON] Training history saved to: {history_json_path.resolve()}")
    except Exception as e:
        print(f"[Warning] Failed to save training history JSON: {e}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "history": history
    }


def main():
    parser = argparse.ArgumentParser(description="Azhar EyeSight – ResNet18 Training Script")
    parser.add_argument("--data_dir", type=str, default=str(config.DATA_DIR), help="Path to data directory")
    parser.add_argument("--save_path", type=str, default=str(config.MODEL_PATH), help="Path to save model checkpoint")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--fine_tune", action="store_true", help="Unfreeze layer4 for domain fine-tuning")

    args = parser.parse_args()

    try:
        run_training(
            data_dir=Path(args.data_dir),
            model_save_path=Path(args.save_path),
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            fine_tune=args.fine_tune
        )
    except Exception as e:
        print(f"\n[Training Terminated] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
