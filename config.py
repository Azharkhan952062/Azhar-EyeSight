"""
Azhar EyeSight - Configuration Module
Centralized configuration parameters for data, model architecture, training, evaluation, and application settings.
"""

from pathlib import Path
import os
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent

# Ensure writable matplotlib cache
CACHE_DIR = BASE_DIR / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "eyesight_resnet18.pth"

RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.png"
PER_CLASS_METRICS_PATH = RESULTS_DIR / "per_class_metrics.png"

ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Target disease classes
CLASSES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

CLASS_LABELS = {
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal"
}

CLASS_DESCRIPTIONS = {
    "cataract": (
        "Clouding of the natural eye lens causing progressive loss of vision. "
        "Characterized by opacification and blurriness in fundus illumination."
    ),
    "diabetic_retinopathy": (
        "Microvascular complication of diabetes causing retinal damage. "
        "Fundus findings include microaneurysms, hemorrhages, hard exudates, and cotton-wool spots."
    ),
    "glaucoma": (
        "Progressive optic neuropathy associated with elevated intraocular pressure. "
        "Characterized by increased optic disc cup-to-disc ratio (cupping) and neuroretinal rim thinning."
    ),
    "normal": (
        "Retina presenting healthy vascular architecture, clear macula, and normal optic disc margin."
    )
}

NUM_CLASSES = len(CLASSES)

# Model architecture & input parameters
MODEL_NAME = "resnet18"
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 0  # 0 is recommended for cross-platform compatibility on Windows

# Hardware accelerator detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical and Educational Disclaimers
MEDICAL_DISCLAIMER = (
    "Azhar EyeSight is an educational and research prototype developed strictly for "
    "scientific demonstration, machine learning education, and algorithm exploration. "
    "It is NOT a certified medical diagnostic device and MUST NOT be used for clinical decision-making. "
    "Model confidence does not equal clinical certainty. Always consult a licensed ophthalmologist "
    "or eye-care professional for qualified medical evaluation and diagnostic screening."
)
