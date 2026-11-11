"""
Azhar EyeSight - Page 3: Model Performance Metrics
Displays verified, empirical evaluation results from the independent test dataset.
Strictly adheres to scientific integrity: never displays fabricated or placeholder numbers.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from utils.metrics import load_metrics_json

st.set_page_config(page_title="Model Performance – Azhar EyeSight", page_icon="📊", layout="wide")

st.title("📊 Model Performance & Evaluation Metrics")
st.markdown("### Empirical Test Set Results on ResNet18 Architecture")
st.markdown("---")

# Load evaluation results
metrics = load_metrics_json(config.METRICS_JSON_PATH)

if metrics is None:
    # Evaluation has not been conducted yet
    st.info("ℹ️ **Evaluation Status: Pending / Not Yet Executed**")
    
    st.markdown(
        """
        In strict adherence to machine learning and scientific integrity standards, **no mock, placeholder,
        or invented performance numbers are displayed on this dashboard**.

        To generate verified performance metrics for your trained ResNet18 model on the test dataset:

        #### Step-by-Step Instructions:
        1. Ensure your independent test images are located inside `data/test/`:
           - `data/test/cataract/`
           - `data/test/diabetic_retinopathy/`
           - `data/test/glaucoma/`
           - `data/test/normal/`
        2. Ensure the trained model weights exist at `models/eyesight_resnet18.pth` (or run `python train.py`).
        3. Execute the evaluation script from your terminal:
        """
    )
    st.code("python evaluate.py", language="bash")
    st.markdown(
        """
        4. Once completed, `evaluate.py` will generate:
           - Quantitative metrics JSON: `results/evaluation_metrics.json`
           - Confusion Matrix Heatmap: `results/confusion_matrix.png`
           - Per-Class Performance Chart: `results/per_class_metrics.png`
        5. Refresh this page to view the empirical evaluation dashboard.
        """
    )

    # Dataset status overview
    st.markdown("---")
    st.subheader("Current Dataset Split Summary")
    from utils.preprocessing import validate_dataset_structure
    report = validate_dataset_structure()
    
    rows = []
    for split in ["train", "val", "test"]:
        split_dict = report["splits"].get(split, {})
        for cls in config.CLASSES:
            rows.append({
                "Dataset Split": split.capitalize(),
                "Disease Category": config.CLASS_LABELS.get(cls, cls),
                "Verified Image Count": split_dict.get(cls, 0)
            })
    df_data = pd.DataFrame(rows)
    st.dataframe(df_data, use_container_width=True, hide_index=True)

else:
    # Metrics exist - display actual verified results
    st.success("✅ **Verified Evaluation Results Loaded from `results/evaluation_metrics.json`**")
    
    acc = metrics.get("accuracy", 0.0)
    prec = metrics.get("macro_precision", 0.0)
    rec = metrics.get("macro_recall", 0.0)
    f1 = metrics.get("macro_f1", 0.0)
    total_samples = metrics.get("total_samples", 0)

    # Top KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Test Accuracy", f"{acc * 100:.2f}%")
    c2.metric("Macro Precision", f"{prec * 100:.2f}%")
    c3.metric("Macro Recall", f"{rec * 100:.2f}%")
    c4.metric("Macro F1-Score", f"{f1 * 100:.2f}%")
    c5.metric("Total Test Images", f"{total_samples}")

    st.markdown("---")

    # Visualizations section
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("Confusion Matrix Heatmap")
        if config.CONFUSION_MATRIX_PATH.is_file():
            st.image(str(config.CONFUSION_MATRIX_PATH), caption="Test Confusion Matrix", use_container_width=True)
        else:
            st.caption("Confusion matrix plot file not found.")

    with col_vis2:
        st.subheader("Per-Class Performance Metrics")
        if config.PER_CLASS_METRICS_PATH.is_file():
            st.image(str(config.PER_CLASS_METRICS_PATH), caption="Precision, Recall & F1 per Class", use_container_width=True)
        else:
            st.caption("Per-class chart file not found.")

    # Per-Class Table Breakdown
    st.markdown("---")
    st.subheader("Detailed Diagnostic Metrics by Category")
    per_class = metrics.get("per_class", {})
    if per_class:
        table_rows = []
        for cls, d in per_class.items():
            table_rows.append({
                "Disease Category": config.CLASS_LABELS.get(cls, cls),
                "Precision (%)": f"{d.get('precision', 0.0) * 100:.2f}%",
                "Recall (%)": f"{d.get('recall', 0.0) * 100:.2f}%",
                "F1-Score (%)": f"{d.get('f1_score', 0.0) * 100:.2f}%",
                "Test Support (Images)": d.get("support", 0)
            })
        st.table(pd.DataFrame(table_rows))

    # Raw Classification Report
    with st.expander("📄 Full Scikit-Learn Classification Report"):
        st.text(metrics.get("classification_report_str", "Report text unavailable."))

    # Training History Plot (if available)
    if config.TRAINING_HISTORY_PATH.is_file():
        st.markdown("---")
        st.subheader("Training and Validation Curves")
        st.image(str(config.TRAINING_HISTORY_PATH), caption="Loss & Accuracy across Epochs", use_container_width=True)

st.markdown("---")
st.markdown(
    f"""
    <div style="background-color: #fff8f1; border-left: 5px solid #ea580c; padding: 14px 18px; border-radius: 6px; font-size: 0.88rem; color: #7c2d12;">
        <b>⚠️ MEDICAL DISCLAIMER:</b> {config.MEDICAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True
)
