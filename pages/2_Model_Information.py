"""
Azhar EyeSight - Page 2: Model Information
Deep Dive into Deep Learning, Convolutional Neural Networks, ResNet18 Architecture,
Transfer Learning, Image Preprocessing, and Explainable AI (Grad-CAM).
"""

import sys
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

st.set_page_config(page_title="Model Information – Azhar EyeSight", page_icon="🧠", layout="wide")

st.title("🧠 Deep Learning & Model Architecture")
st.markdown("### Technical Foundations of Azhar EyeSight's Neural Vision Pipeline")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 What is a CNN?",
    "🏗️ ResNet18 Architecture",
    "🔄 Transfer Learning",
    "⚙️ Preprocessing & Augmentation",
    "🔬 Explainable AI (Grad-CAM)"
])

with tab1:
    st.subheader("What is a Convolutional Neural Network (CNN)?")
    st.markdown(
        """
        A **Convolutional Neural Network (CNN)** is a class of artificial neural networks specifically engineered
        for processing structured grid data, such as images. Unlike traditional fully connected neural networks
        where every input pixel connects to every neuron—causing exponential parameter explosion and losing spatial
        geometry—CNNs leverage two core principles:

        1. **Local Receptive Fields**: Small learnable parameter kernels (e.g., 3×3 or 7×7 matrices) slide over the
           image, computing dot products to detect localized features (such as contrast boundaries, edges, and blood vessel curves).
        2. **Weight Sharing & Translation Invariance**: The same kernel weights are reused across the entire image canvas.
           If a microaneurysm or retinal hemorrhage appears in the upper quadrant, the feature filter detects it with the same
           efficiency as in the lower quadrant.
        
        As layers deepen:
        - **Early Layers**: Learn low-level visual primitives (edges, color gradients, texture lines).
        - **Middle Layers**: Assemble primitives into structural motifs (vessel branching patterns, optic disc rims).
        - **Deep Layers**: Recognize high-level pathological landmarks (exudates, optic cup cupping, lens clouding).
        """
    )

with tab2:
    st.subheader("ResNet18: Residual Networks")
    st.markdown(
        """
        ### The Degradation Problem
        In classical convolutional networks (e.g., AlexNet, VGG), adding more layers beyond a certain depth led to the
        **degradation problem**: as network depth increased, accuracy saturated and then rapidly degraded, caused by vanishing
        or exploding gradients during backpropagation.
        
        ### The Residual Skip Connection Solution
        Introduced by He et al. (2015), **ResNet (Residual Network)** solved this by introducing **identity shortcut connections**
        (skip connections). Instead of forcing stacked layers to fit an underlying mapping \\( H(x) \\), the layers are trained
        to fit a residual mapping:
        """
    )
    st.latex(r"F(x) = H(x) - x \implies H(x) = F(x) + x")
    st.markdown(
        """
        If an identity mapping is optimal, the network can easily drive the residual weights \\( F(x) \\to 0 \\), allowing gradients
        to flow directly through the identity pathway during backpropagation without attenuation.

        ### ResNet18 Layer Breakdown in Azhar EyeSight:
        - **Initial Conv Block**: 7×7 convolution, stride 2, 64 channels, followed by BatchNorm, ReLU, and 3×3 Max Pooling.
        - **Layer 1**: 2 Residual blocks (64 feature channels).
        - **Layer 2**: 2 Residual blocks with downsampling (128 channels).
        - **Layer 3**: 2 Residual blocks with downsampling (256 channels).
        - **Layer 4**: 2 Residual blocks with downsampling (512 channels) — **Target layer for Grad-CAM**.
        - **Global Average Pooling (GAP)**: Collapses each 512-channel feature map from \\( 7 \times 7 \\) to a single average \\( 1 \times 1 \\).
        - **Modified Linear Head**: Replaced original 1000-class ImageNet output with a 4-unit linear layer:
        """
    )
    st.code("model.fc = nn.Linear(in_features=512, out_features=4)", language="python")

with tab3:
    st.subheader("Transfer Learning Paradigm")
    st.markdown(
        """
        Training a deep convolutional network from scratch requires hundreds of thousands of carefully annotated medical
        images and weeks of high-performance GPU compute. 
        
        **Transfer Learning** circumvents this data scarcity by transferring visual representations learned on **ImageNet**
        (1.2 million natural images across 1000 categories) to the specialized task of retinal fundus classification.

        ### Our Two-Stage Transfer Strategy:
        1. **Feature Extraction (Frozen Backbone)**:
           - All pretrained convolutional layers in `layer1`, `layer2`, `layer3`, and `layer4` have their gradients frozen (`requires_grad = False`).
           - Only the freshly initialized linear classification head (`model.fc`) is trained using the Adam optimizer (learning rate \\( 10^{-4} \\)).
           - This prevents destructive gradient updates from overwriting the general visual feature detectors.
        
        2. **Domain Fine-Tuning (Deep Unfreezing)**:
           - In fine-tuning mode (`--fine_tune`), `layer4` is unfrozen, allowing the deepest feature extractors to fine-tune
             their representations specifically to retinal optic disc margins and microvascular patterns.
        """
    )

with tab4:
    st.subheader("Image Preprocessing & Data Augmentation")
    st.markdown(
        """
        ### Deterministic Preprocessing Pipeline:
        1. **Format & Safety Verification**: File size validation (<25MB), corrupt stream check, EXIF orientation alignment.
        2. **Color Mode Standardization**: Conversion of grayscale, RGBA, or CMYK inputs to 3-channel RGB.
        3. **Bicubic Resizing**: Fixed resolution of \\( 224 \times 224 \\) pixels to match ResNet18 input specifications.
        4. **ImageNet Normalization**: Channel-wise zero-centering using standard statistics:
        """
    )
    st.latex(r"\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]")
    st.latex(r"x_{\text{normalized}} = \frac{x - \mu}{\sigma}")

    st.markdown(
        """
        ### Biologically-Informed Training Augmentations:
        Data augmentation expands dataset diversity and prevents overfitting:
        - **Random Horizontal Flip (p=0.5)**: Retinal fundus anatomy is biologically symmetric between left and right eyes.
        - **Small Random Rotation (±15°)**: Simulates head tilt and varying patient positioning during fundus photography.
        - **Color Jitter (Brightness & Contrast ±10%)**: Accommodates variable camera flash exposure and ocular media clarity.
        - **Strict Isolation**: Augmentations are applied **only** during training; validation and test sets remain unaltered.
        """
    )

with tab5:
    st.subheader("Explainable AI: Grad-CAM Formulation")
    st.markdown(
        """
        Deep neural networks are often criticized as 'black boxes'. **Gradient-weighted Class Activation Mapping (Grad-CAM)**
        provides visual interpretability by illuminating the spatial regions of the image that stimulated a specific decision.

        ### Mathematical Formulation:
        Let \\( y^c \\) be the score (logit before softmax) for target class \\( c \\), and let \\( A^k \\) be the \\( k \\)-th
        feature map of the final convolutional layer (`model.layer4[-1]`).
        
        **1. Importance Weights (\\( \alpha_k^c \\)):**
        We compute the gradients of \\( y^c \\) with respect to feature map \\( A^k \\), and globally average pool across spatial dimensions \\( (i, j) \\):
        """
    )
    st.latex(r"\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}")
    st.markdown(
        """
        **2. Heatmap Construction:**
        We compute a weighted linear combination of the forward feature maps, followed by a Rectified Linear Unit (ReLU):
        """
    )
    st.latex(r"L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)")
    st.markdown(
        """
        The **ReLU** ensures that we capture only features that have a **positive** influence on the target class score,
        filtering out features that correlate with other classes.
        
        The resulting 2D heatmap is normalized between \\( [0, 1] \\), colored with the JET palette, and blended over the
        original retinal image at 45% opacity.
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
