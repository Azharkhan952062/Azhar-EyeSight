"""
Azhar EyeSight - Page 1: About Project
Comprehensive overview of project objectives, clinical problem statement,
system architecture, technology stack, and ethical boundaries.
"""

import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

st.set_page_config(page_title="About Project – Azhar EyeSight", page_icon="📖", layout="wide")

st.title("📖 About Azhar EyeSight")
st.markdown("### AI-Powered Retinal Fundus Image Classification Prototype")
st.markdown("---")

# 1. Project Overview
st.subheader("1. Project Overview")
st.markdown(
    """
    **Azhar EyeSight** is an educational and scientific research prototype designed to demonstrate how
    modern Deep Learning (CNNs), Computer Vision, and Transfer Learning can be integrated to analyze
    digital retinal fundus photographs. 

    The system accepts color fundus photography and evaluates the likelihood of four key ocular states:
    1. **Cataract**
    2. **Diabetic Retinopathy**
    3. **Glaucoma**
    4. **Normal (Healthy Retina)**

    It combines algorithmic transparency via **Grad-CAM (Gradient-weighted Class Activation Mapping)**
    with strict guardrails, making clear distinctions between probabilistic machine learning confidence
    and definitive medical diagnosis.
    """
)

# 2. Problem Statement
st.subheader("2. Problem Statement")
st.markdown(
    """
    According to the World Health Organization (WHO), over 2.2 billion people worldwide live with vision impairment.
    In at least 1 billion of these cases, vision impairment could have been prevented or has yet to be addressed.
    
    The three leading causes of preventable and manageable blindness are:
    - **Cataract**: Opacification of the crystalline lens, causing progressive vision blur. While curable with surgery,
      delays in identification lead to significant quality-of-life reduction.
    - **Diabetic Retinopathy (DR)**: A microvascular complication of diabetes causing retinal capillary occlusion,
      microaneurysms, hemorrhages, and neovascularization. DR is the leading cause of preventable blindness in working-age adults.
    - **Glaucoma**: The 'silent thief of sight', characterized by progressive optic nerve degeneration and optic disc cupping,
      often without noticeable symptoms until severe irreversible peripheral vision loss occurs.
    
    In developing regions and underserved rural communities, the ratio of ophthalmologists to patients can be as low as
    1 per 100,000 citizens. Automated screening tools can serve as educational aids and research frameworks for triaging,
    highlighting cases requiring urgent clinical attention.
    """
)

# 3. Project Objectives
st.subheader("3. Project Objectives")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        - **Accurate Categorization**: Classify 4 distinct ocular conditions with a deep convolutional neural network.
        - **Transfer Learning**: Leverage ImageNet pretrained representations in ResNet18 to optimize convergence.
        - **Interpretability & Explainability**: Integrate Grad-CAM heatmaps to visualize the model's visual attention.
        """
    )
with c2:
    st.markdown(
        """
        - **Production-Style Architecture**: Clean, modular Python codebase with dedicated modules for preprocessing, training, evaluation, and inference.
        - **Responsible AI**: Promote transparent boundaries, displaying clear medical disclaimers and refusing to claim clinical certainty.
        - **Accessible Delivery**: Responsive multi-page web interface built with Streamlit and Plotly.
        """
    )

# 4. System Workflow
st.subheader("4. System Architecture & Workflow")
st.markdown(
    """
    The end-to-end processing pipeline operates through five sequential phases:
    
    ```
    ┌──────────────────────┐
    │ Retinal Fundus Image │  (JPG, PNG, JPEG)
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Safe Preprocessing   │  - Size validation (<25MB) & EXIF orientation fix
    │  & Normalization     │  - Resize to 224x224 RGB
    └──────────┬───────────┘  - ImageNet normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
               │
               ▼
    ┌──────────────────────┐
    │  PyTorch ResNet18    │  - 18 Residual Layers with skip connections
    │  Transfer Learning   │  - Replaced 4-class classification head
    └──────────┬───────────┘
               │
        ┌──────┴───────────────────────────┐
        ▼                                  ▼
    ┌──────────────────────────┐    ┌──────────────────────────┐
    │   Softmax Probabilities  │    │     Grad-CAM Engine      │
    │  - Predicted Category    │    │  - Hook into layer4[-1]  │
    │  - Confidence Metric     │    │  - Gradient-weighted CAM │
    │  - Class Distribution    │    │  - Jet Heatmap Overlay   │
    └──────────┬───────────────┘    └──────────┬───────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Interactive Streamlit Dashboard & Educational Summary   │
    └──────────────────────────────────────────────────────────┘
    ```
    """
)

# 5. Technology Stack
st.subheader("5. Technology Stack")
tech_data = {
    "Domain": ["Deep Learning", "Vision Backbone", "Computer Vision", "Machine Learning", "Frontend & UI", "Data Serialization", "Data Engine"],
    "Tool / Library": ["PyTorch 2.x", "Torchvision (ResNet18)", "OpenCV & PIL (Pillow)", "Scikit-learn", "Streamlit", "JSON & Matplotlib", "NumPy & Pandas"],
    "Role in Azhar EyeSight": [
        "Tensor computation, autograd, forward/backward execution",
        "Pretrained weights, residual blocks, layer4 feature extraction",
        "Image format validation, resizing, color channel conversion, Grad-CAM blending",
        "Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix",
        "Modern healthcare dashboard, multi-page layout, interactive Plotly visualizations",
        "Confusion matrix generation, training history curves, evaluation metrics persistence",
        "Array operations, data structures, tabular metrics display"
    ]
}
st.table(tech_data)

# 6. Ethical Scope and Limitations
st.subheader("6. Clinical Limitations & Ethical Scope")
st.warning(
    """
    **Known Limitations:**
    - **Single-Modality Input**: Clinical ophthalmology relies on multimodal evaluation (OCT scans, visual field perimeter testing, fundus autofluorescence, intraocular pressure measurement). Fundus photographs alone cannot provide complete diagnostic certainty.
    - **Dataset Demographics & Bias**: Pretrained algorithms may reflect distribution shifts based on fundus camera hardware, lighting variations, pupil dilation status, and ethnic pigmentation differences in retinal backgrounds.
    - **Not a Diagnostic Tool**: Azhar EyeSight is not certified by FDA, CE, or CDSCO as a medical device.
    """
)

st.markdown("---")
st.markdown(
    f"""
    <div style="background-color: #fff8f1; border-left: 5px solid #ea580c; padding: 14px 18px; border-radius: 6px; font-size: 0.88rem; color: #7c2d12;">
        <b>⚠️ MEDICAL DISCLAIMER:</b> {config.MEDICAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True
)
