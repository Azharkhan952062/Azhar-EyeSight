# Azhar EyeSight – AI-Based Eye Disease Classification System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-ResNet18-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Azhar EyeSight** is an educational, production-style deep learning application designed to classify digital retinal fundus photographs into four distinct ocular diagnostic categories:

1. **Cataract**
2. **Diabetic Retinopathy**
3. **Glaucoma**
4. **Normal (Healthy Retina)**

Built using **PyTorch**, **Torchvision (ResNet18 Transfer Learning)**, **OpenCV**, **Scikit-learn**, and a modern healthcare-inspired **Streamlit** user interface featuring explainable AI via **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

> [!WARNING]
> **MANDATORY MEDICAL & RESEARCH DISCLAIMER**:
> Azhar EyeSight is an educational and research prototype developed strictly for academic demonstration, computer vision exploration, and machine learning education. It is **NOT** a certified medical diagnostic device and **MUST NOT** be used for clinical decision-making. Model confidence scores are mathematical likelihoods and do not equal clinical certainty. Always consult a licensed ophthalmologist or board-certified physician for professional eye evaluation and diagnosis.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Technology Stack](#technology-stack)
3. [Project Architecture](#project-architecture)
4. [Dataset Setup Guide](#dataset-setup-guide)
5. [Installation & Environment Setup](#installation--environment-setup)
6. [Model Training](#model-training)
7. [Model Evaluation & Metrics](#model-evaluation--metrics)
8. [CLI Prediction Engine](#cli-prediction-engine)
9. [Running the Streamlit Web Application](#running-the-streamlit-web-application)
10. [Streamlit Cloud & GitHub Deployment Guide](#deployment-guide)
11. [Clinical Limitations & Responsible AI](#clinical-limitations)
12. [Future Improvements](#future-improvements)

---

## Key Features

- **Deep Transfer Learning**: Utilizes ImageNet-pretrained ResNet18 with custom 4-class linear classification head.
- **Biologically-Informed Preprocessing**: ImageNet standard normalization, RGB standardization, and symmetric augmentations (horizontal flips, rotation, exposure adjustments).
- **Explainable AI (XAI)**: Native PyTorch hook-based **Grad-CAM** generates activation heatmaps highlighting salient retinal regions that drive network decisions.
- **Production-Style Modularity**: Clean separation of concerns with dedicated modules for configuration, dataset handling, architecture, evaluation metrics, visualization, and inference.
- **Comprehensive Evaluation**: Automated generation of Confusion Matrices, Accuracy, Macro/Weighted Precision, Recall, F1-scores, and per-class diagnostic reports using Scikit-learn.
- **Responsive Healthcare UI**: Multi-page Streamlit application styled with custom CSS cards, status indicators, and interactive Plotly probability distributions.
- **Zero Mock Data Policy**: Does not display fabricated performance figures; clearly displays pending status until real evaluation is executed on test data.

---

## Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Core Language** | Python 3.10 / 3.11 / 3.13 | Primary programming runtime |
| **Deep Learning** | PyTorch 2.x | Neural network creation, Autograd, tensor computation |
| **Vision Backbone** | Torchvision ResNet18 | Pretrained feature extractor with residual skip connections |
| **Computer Vision** | OpenCV & Pillow (PIL) | Safe image decoding, EXIF orientation, colormap blending |
| **Machine Learning** | Scikit-learn | Precision, Recall, F1, Confusion Matrix, Classification Report |
| **Visualization** | Matplotlib, Seaborn, Plotly | High-resolution publication plots & interactive charts |
| **Web Frontend** | Streamlit | Healthcare-styled dashboard and multi-page application |
| **Data Engine** | NumPy & Pandas | Numerical matrix manipulation and tabular metrics display |

---

## Project Architecture

```
Azhar-EyeSight/
├── app.py                     # Primary Streamlit Dashboard entry point
├── train.py                   # PyTorch ResNet18 training pipeline
├── evaluate.py                # Test set evaluation and metric generation
├── predict.py                 # Standalone prediction and Grad-CAM CLI
├── setup_dataset.py           # Dataset audit and split organization utility
├── config.py                  # Centralized configuration and constants
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Version control exclusions
│
├── models/                    # Saved PyTorch checkpoints
│   ├── .gitkeep
│   └── eyesight_resnet18.pth  # Trained model weights (generated via train.py)
│
├── data/                      # Dataset splits (patient-isolated)
│   ├── train/                 # Training set (cataract, diabetic_retinopathy, glaucoma, normal)
│   ├── val/                   # Validation set
│   └── test/                  # Independent hold-out test set
│
├── results/                   # Evaluation artifacts
│   ├── .gitkeep
│   ├── evaluation_metrics.json
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── training_history.png
│
├── utils/                     # Modular backend utilities
│   ├── __init__.py
│   ├── preprocessing.py       # Image validation, transforms, dataset auditing
│   ├── model.py               # ResNet18 builder, layer freezing/unfreezing, checkpoints
│   ├── visualization.py       # Grad-CAM engine, heatmap overlays, Matplotlib & Plotly charts
│   └── metrics.py             # Scikit-learn metric calculations and serialization
│
├── pages/                     # Streamlit multi-page interface
│   ├── 1_About_Project.py     # Background, problem statement, workflow
│   ├── 2_Model_Information.py # CNN, ResNet18, Transfer Learning, Grad-CAM math
│   └── 3_Performance_Metrics.py # Empirical evaluation dashboard & reports
│
└── assets/
    └── logo.png               # High-resolution Azhar EyeSight retina logo
```

---

## Dataset Setup Guide

Azhar EyeSight supports four target classes:
- `cataract`
- `diabetic_retinopathy`
- `glaucoma`
- `normal`

### Recommended Public Retinal Fundus Datasets
1. **ODIR-5K (Ocular Disease Intelligent Recognition)**: Public challenge dataset containing fundus photos labeled for Cataract, Glaucoma, Diabetic Retinopathy, and Normal.
2. **Kaggle Eye Disease Classification Datasets**:
   - [Ocular Disease Recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
   - [Retinal Disease Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
3. **Mendeley Retinal Fundus Multi-Disease Database**.

### Expected Directory Hierarchy
Place retinal images in the corresponding split directories:

```
data/
  train/
    cataract/
    diabetic_retinopathy/
    glaucoma/
    normal/
  val/
    cataract/
    diabetic_retinopathy/
    glaucoma/
    normal/
  test/
    cataract/
    diabetic_retinopathy/
    glaucoma/
    normal/
```

### Verifying or Automatically Splitting Data
To audit your current dataset structure at any time:
```bash
python setup_dataset.py --status
```

If you have a raw un-split folder with subfolders for each class, you can automatically split it into 70% train, 15% validation, and 15% test:
```bash
python setup_dataset.py --source /path/to/raw_dataset --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15
```

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/azhar-eyesight.git
cd azhar-eyesight
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Model Training

The training script trains a ResNet18 convolutional neural network using CrossEntropyLoss and Adam optimizer, featuring learning rate decay and early stopping.

### Default Feature Extraction Training (Backbone Frozen, Head Trained)
```bash
python train.py --epochs 15 --batch_size 16 --lr 0.0001 --patience 4
```

### Fine-Tuning Mode (Unfreezes ResNet18 Layer4 for Deep Domain Adaptation)
```bash
python train.py --epochs 20 --batch_size 16 --lr 0.00005 --fine_tune
```

### Command Line Arguments
- `--data_dir`: Root directory containing `train/` and `val/` (default: `data/`)
- `--save_path`: Path to save the best model checkpoint (default: `models/eyesight_resnet18.pth`)
- `--batch_size`: Batch size for DataLoaders (default: `16`)
- `--lr`: Learning rate (default: `0.0001`)
- `--epochs`: Maximum number of training epochs (default: `15`)
- `--patience`: Early stopping patience (default: `4`)
- `--fine_tune`: Flag to unfreeze `layer4` for domain-specific fine-tuning

---

## Model Evaluation & Metrics

To evaluate your trained model against the independent hold-out test set (`data/test/`):

```bash
python evaluate.py --model models/eyesight_resnet18.pth --test_dir data/test
```

### Evaluation Artifacts Generated:
- `results/evaluation_metrics.json`: Full quantitative report (Accuracy, Macro/Weighted F1, Precision, Recall, Support).
- `results/confusion_matrix.png`: Annotated diagnostic confusion matrix heatmap.
- `results/per_class_metrics.png`: Grouped comparative bar chart of Precision, Recall, and F1 per disease.

---

## CLI Prediction Engine

You can run individual retinal image inferences directly from your command line:

```bash
# Standard prediction
python predict.py --image path/to/retinal_image.jpg

# Prediction with Grad-CAM explainability heatmap saved
python predict.py --image path/to/retinal_image.jpg --gradcam --save_gradcam results/gradcam_sample.jpg
```

---

## Running the Streamlit Web Application

To launch the full interactive web application:

**Windows / Linux / macOS:**
```bash
streamlit run app.py
```

Open your browser to: **`http://localhost:8501`**

### Application Pages:
- **Main Dashboard (`app.py`)**: Drag-and-drop fundus image upload, real-time prediction, confidence scores, Plotly probability breakdown, and interactive Grad-CAM heatmap visualization.
- **Page 1 – About Project (`pages/1_About_Project.py`)**: Clinical problem statement, epidemiology of ocular diseases, system workflow, and technology overview.
- **Page 2 – Model Information (`pages/2_Model_Information.py`)**: Detailed technical dive into CNNs, ResNet18 residual learning ($F(x) + x$), Transfer Learning, data augmentations, and the mathematical derivation of Grad-CAM.
- **Page 3 – Performance Metrics (`pages/3_Performance_Metrics.py`)**: Dynamic display of verified empirical test metrics, confusion matrix, and per-class performance tables.

---

## Deployment Guide

### Deploying to GitHub

1. Initialize Git and stage files:
```bash
git init
git add .
git commit -m "feat: Initial commit for Azhar EyeSight AI classification system"
```

2. Link to your GitHub repository and push:
```bash
git branch -M main
git remote add origin https://github.com/your-username/azhar-eyesight.git
git push -u origin main
```

> **Model weights note:** `models/eyesight_resnet18.pth` (~43 MB) is tracked by Git and will
> be included in your commit. This is within GitHub's 100 MB per-file limit and is the
> simplest way to make predictions available on Streamlit Cloud.

### Deploying to Streamlit Community Cloud

1. Push your repository `azhar-eyesight` to GitHub (see steps above).
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `your-username/azhar-eyesight`.
5. Set the **Main file path** to: `app.py`.
6. Click **"Advanced settings"** and set the **Python version** to `3.11`.
7. Target App URL: `https://azhar-eyesight.streamlit.app` (subject to availability).
8. Click **"Deploy"**.

> **Important:** The trained model weights (`models/eyesight_resnet18.pth`) must be present
> in the repository for the app to make predictions. If you retrain locally and generate a
> new `.pth` file, commit and push it to GitHub before redeploying.

---

## Clinical Limitations

1. **Single-Modality Input**: Clinical ophthalmology evaluates patients through multimodal testing—including Optical Coherence Tomography (OCT), intraocular pressure (tonometry), slit-lamp exams, and visual field perimetry. Fundus photographs alone cannot provide a definitive diagnosis.
2. **Dataset Demographics & Equipment Variance**: Fundus cameras differ in illumination optics, aperture, field of view (30° vs. 45° vs. 200° ultra-widefield), and pigmentation appearance across ethnicities.
3. **Research Prototype**: This software is not FDA 510(k) cleared or CE mark certified.

---

## Future Improvements

- [ ] Multi-head architecture supporting combined optical coherence tomography (OCT) and color fundus fusion.
- [ ] Integration with Vision Transformers (ViT / Swin Transformer) for global contextual attention.
- [ ] Automated image quality assessment (identifying underexposed or blurred fundus photos before inference).
- [ ] Deployment of lightweight ONNX runtime or TensorRT for low-latency clinical edge devices.

---

## License & Credits

Developed as an educational research prototype. Built with PyTorch and Streamlit.
Logo and branding assets designed for **Azhar EyeSight**.
