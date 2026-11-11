"""
Azhar EyeSight – AI-Based Eye Disease Classification System
Main Streamlit Application Dashboard
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from utils.preprocessing import safe_load_image
from utils.visualization import create_probability_chart
from predict import EyeDiseasePredictor

# Page Configuration
st.set_page_config(
    page_title="Azhar EyeSight – Retinal AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Styling CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header branding container */
    .brand-container {
        background: linear-gradient(135deg, #0b2545 0%, #134074 50%, #0077b6 100%);
        padding: 28px 32px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(11, 37, 69, 0.12);
    }
    
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #ffffff;
    }
    
    .brand-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #caf0f8;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        background-color: rgba(255, 255, 255, 0.18);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Healthcare Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    
    /* Disease Category Badges */
    .badge-cataract {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .badge-diabetic_retinopathy {
        background-color: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .badge-glaucoma {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .badge-normal {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Medical Disclaimer Banner */
    .disclaimer-box {
        background-color: #fff8f1;
        border-left: 5px solid #ea580c;
        padding: 16px 20px;
        border-radius: 6px;
        margin-top: 24px;
        font-size: 0.88rem;
        color: #7c2d12;
        line-height: 1.5;
    }
    
    .disclaimer-title {
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar():
    """Renders the left navigation sidebar with system health and parameters."""
    with st.sidebar:
        # Logo display
        if config.LOGO_PATH.is_file():
            st.image(str(config.LOGO_PATH), width=160)
        else:
            st.title("👁️ Azhar EyeSight")

        st.caption("AI Retinal Disease Analysis System")
        st.markdown("---")

        # System Status
        st.markdown("### 🖥️ System Status")
        
        # Check model weights
        model_exists = config.MODEL_PATH.is_file()
        if model_exists:
            st.success("✅ Model: Loaded (ResNet18)", icon="🟢")
        else:
            st.warning("⚠️ Model: Weights Missing", icon="🟠")

        # Hardware Accelerator
        device_name = "CUDA (GPU Accelerator)" if config.DEVICE.type == "cuda" else "CPU (Standard)"
        st.info(f"Hardware: **{device_name}**")

        st.markdown("---")
        st.markdown("### 📋 Supported Diagnoses")
        for cls_name, label in config.CLASS_LABELS.items():
            st.markdown(f"- **{label}**")

        st.markdown("---")
        st.markdown("### ⚕️ Clinical Disclaimer")
        st.caption(
            "Azhar EyeSight is an educational research tool, NOT a certified medical device. "
            "Model outputs are mathematical predictions and cannot substitute for clinical examination "
            "by a licensed ophthalmologist."
        )


def main():
    render_sidebar()

    # Main Brand Header
    st.markdown(
        """
        <div class="brand-container">
            <h1 class="brand-title">Azhar EyeSight</h1>
            <div class="brand-subtitle">AI-Based Retinal Image Classification using Deep Learning & Computer Vision</div>
            <div>
                <span class="badge-pill">PyTorch ResNet18</span>
                <span class="badge-pill">Transfer Learning</span>
                <span class="badge-pill">Explainable AI (Grad-CAM)</span>
                <span class="badge-pill">4 Retinal Classes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check if model exists
    model_exists = config.MODEL_PATH.is_file()
    if not model_exists:
        st.warning(
            f"**Trained model weights not found at `{config.MODEL_PATH.name}`.**\n\n"
            "The application interface is fully operational. To perform live inferences:\n"
            "1. Place your dataset inside `data/train/`, `data/val/`, and `data/test/`.\n"
            "2. Run model training: `python train.py`\n"
            "3. The best model will automatically be saved to `models/eyesight_resnet18.pth`.",
            icon="ℹ️"
        )

    # Main Grid Layout: Left Column = Input & Preview, Right Column = Diagnostics & Visual Analysis
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Retinal Fundus Image Upload")
        st.markdown(
            "Upload a digital retinal fundus photograph (`.jpg`, `.jpeg`, `.png`). "
            "Ensure the optic disc, macula, and retinal vascular arcade are centered."
        )

        uploaded_file = st.file_uploader(
            "Choose a fundus image...",
            type=["jpg", "jpeg", "png"],
            help="High-resolution color fundus photographs recommended (max 25MB)."
        )

        gradcam_toggle = st.checkbox(
            "Enable Grad-CAM Visual Attention Heatmap",
            value=True,
            help="Highlights the salient retinal features influencing the neural network prediction."
        )

        if uploaded_file is not None:
            try:
                pil_image = safe_load_image(uploaded_file)
                st.image(pil_image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
            except Exception as e:
                st.error(f"Image validation failed: {e}")
                pil_image = None
        else:
            pil_image = None
            st.info("👆 Upload a retinal fundus image above to begin diagnostic classification.")

    with col_results:
        st.subheader("2. AI Diagnostic Prediction")

        if uploaded_file is not None and pil_image is not None:
            analyze_button = st.button("🔍 Analyze Retinal Image", type="primary", use_container_width=True)

            if analyze_button or "last_prediction" in st.session_state:
                if not model_exists:
                    st.error(
                        "Cannot run inference because trained weights were not found. "
                        "Please train the model via `python train.py` first."
                    )
                else:
                    with st.spinner("Analyzing retinal structures and calculating class activation maps..."):
                        try:
                            # Load predictor and run prediction
                            predictor = EyeDiseasePredictor(model_path=config.MODEL_PATH)
                            result = predictor.predict(pil_image, generate_gradcam=gradcam_toggle)
                            st.session_state["last_prediction"] = result
                        except Exception as e:
                            st.error(f"Inference error: {e}")
                            result = None

                    if result is not None:
                        pred_class = result["predicted_class"]
                        pred_label = result["predicted_label"]
                        confidence = result["confidence"]
                        badge_class = f"badge-{pred_class}"

                        # Prediction Summary Cards
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-label">Predicted Condition</div>
                                    <div style="margin-top: 8px;">
                                        <span class="{badge_class}" style="font-size: 1.15rem;">{pred_label}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        with c2:
                            st.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-label">Model Confidence</div>
                                    <div class="metric-value">{confidence * 100:.1f}%</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Softmax Probability Distribution Chart
                        st.markdown("#### Class Probability Distribution")
                        prob_fig = create_probability_chart(result["probabilities"])
                        st.plotly_chart(prob_fig, use_container_width=True)

                        # Clinical Context Guide
                        with st.expander("ℹ️ Clinical Description & Pathological Findings", expanded=True):
                            st.markdown(f"**About {pred_label}:**")
                            st.write(result["description"])
                            st.markdown(
                                "*Note: A definitive diagnosis requires multimodal testing, including optical coherence "
                                "tomography (OCT), intraocular pressure tonometry, slit-lamp biomicroscopy, and formal visual field testing.*"
                            )

                        # Grad-CAM Visualization Display
                        if gradcam_toggle and result.get("gradcam_overlay") is not None:
                            st.markdown("#### 🔬 Explainable AI: Grad-CAM Retinal Attention")
                            st.caption(
                                "Gradient-weighted Class Activation Mapping (Grad-CAM) illuminates the receptive "
                                "field regions in the final convolutional layer (layer4) that contributed most to this classification."
                            )

                            v1, v2 = st.columns(2)
                            with v1:
                                st.image(result["original_image"], caption="Original Retinal Input", use_container_width=True)
                            with v2:
                                st.image(result["gradcam_overlay"], caption=f"Grad-CAM Heatmap ({pred_label})", use_container_width=True)
                            
                            st.caption(
                                "⚠️ *Grad-CAM reflects statistical correlations learned by the neural network during training, "
                                "NOT proof of verified clinical reasoning or histological pathology.*"
                            )

        else:
            st.markdown(
                """
                <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 48px 24px; text-align: center; color: #64748b;">
                    <h3 style="color: #475569; margin-bottom: 8px;">No Retinal Image Uploaded</h3>
                    <p style="margin: 0; font-size: 0.95rem;">Please select and upload a fundus photograph on the left panel to execute real-time AI classification.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Prominent Medical Disclaimer Banner at bottom
    st.markdown("---")
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <div class="disclaimer-title">⚠️ MANDATORY MEDICAL & RESEARCH DISCLAIMER</div>
            <div>{config.MEDICAL_DISCLAIMER}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
