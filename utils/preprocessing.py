"""
Azhar EyeSight - Image Preprocessing and Dataset Utilities
Provides image validation, PyTorch transforms, safe image loading, and dataset verification.
"""

import io
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
from PIL import Image, ImageOps
import torch
from torchvision import transforms

import config

# Allowed file extensions for retinal images
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


def get_train_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """
    Builds data augmentation and preprocessing pipeline for training.
    
    Includes:
    - Resize to target dimension (224x224)
    - Random Horizontal Flip (retinal fundus images are biologically symmetric)
    - Small Random Rotation (+/- 15 degrees)
    - Gentle brightness and contrast jitter to simulate variable fundus camera exposure
    - Conversion to PyTorch Tensor
    - Normalization with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def get_val_test_transforms(image_size: Tuple[int, int] = config.IMAGE_SIZE) -> transforms.Compose:
    """
    Deterministic preprocessing pipeline for validation and evaluation.
    No random augmentations are applied.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def safe_load_image(image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> Image.Image:
    """
    Safely loads and validates a retinal fundus image from various input types.
    
    - Validates file size and format.
    - Resolves EXIF orientation tags.
    - Converts any color mode (RGBA, Grayscale, CMYK) to standard 3-channel RGB.
    
    Raises:
        ValueError: If file format is unsupported, size exceeds limit, or image is corrupt.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise ValueError(f"Image file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format '{path.suffix}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        if path.stat().st_size > MAX_IMAGE_FILE_SIZE_BYTES:
            raise ValueError(
                f"Image file size ({path.stat().st_size / (1024*1024):.1f} MB) exceeds maximum allowed limit of 25 MB."
            )
        try:
            img = Image.open(path)
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file: {e}")
    elif isinstance(image_input, (bytes, io.BytesIO)):
        bio = image_input if isinstance(image_input, io.BytesIO) else io.BytesIO(image_input)
        # Check size if bytes
        bio.seek(0, io.SEEK_END)
        size = bio.tell()
        bio.seek(0)
        if size > MAX_IMAGE_FILE_SIZE_BYTES:
            raise ValueError(f"Image payload size ({size / (1024*1024):.1f} MB) exceeds maximum 25 MB limit.")
        try:
            img = Image.open(bio)
        except Exception as e:
            raise ValueError(f"Failed to decode uploaded image data: {e}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    try:
        # Verify image integrity
        img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted image stream: {e}")

    # Re-open after verify() because verify() invalidates the file pointer
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
    elif isinstance(image_input, (bytes, io.BytesIO)):
        bio.seek(0)
        img = Image.open(bio)
    elif isinstance(image_input, Image.Image):
        img = image_input.copy()

    # Transpose according to EXIF orientation if present
    img = ImageOps.exif_transpose(img)

    # Ensure 3-channel RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def preprocess_image_for_inference(
    image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    image_size: Tuple[int, int] = config.IMAGE_SIZE
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Full inference preprocessing pipeline.
    
    Returns:
        tuple: (batch_tensor, pil_rgb_image)
        where batch_tensor has shape [1, 3, 224, 224] ready for model input.
    """
    pil_image = safe_load_image(image_input)
    val_transform = get_val_test_transforms(image_size)
    tensor = val_transform(pil_image)
    batch_tensor = tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]
    return batch_tensor, pil_image


def validate_dataset_structure(
    data_dir: Union[str, Path] = config.DATA_DIR,
    classes = config.CLASSES
) -> Dict[str, Any]:
    """
    Scans and verifies the data directory hierarchy for train, val, and test splits.
    
    Returns a status dictionary detailing image counts per class and any missing folders.
    """
    data_path = Path(data_dir)
    splits = ["train", "val", "test"]
    report = {
        "is_valid": True,
        "splits": {},
        "missing_folders": [],
        "empty_classes": [],
        "total_images": 0
    }

    if not data_path.is_dir():
        report["is_valid"] = False
        report["error"] = f"Root data directory does not exist: {data_path}"
        return report

    for split in splits:
        split_dir = data_path / split
        report["splits"][split] = {}
        split_count = 0

        if not split_dir.is_dir():
            report["is_valid"] = False
            report["missing_folders"].append(str(split_dir))
            continue

        for cls in classes:
            class_dir = split_dir / cls
            if not class_dir.is_dir():
                report["is_valid"] = False
                report["missing_folders"].append(str(class_dir))
                report["splits"][split][cls] = 0
                continue

            # Count valid images in class folder
            valid_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            count = len(valid_files)
            report["splits"][split][cls] = count
            split_count += count

            if count == 0:
                report["empty_classes"].append(f"{split}/{cls}")

        report["total_images"] += split_count

    if report["empty_classes"] or report["missing_folders"]:
        report["is_valid"] = False

    return report
