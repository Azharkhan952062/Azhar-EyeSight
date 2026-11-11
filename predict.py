"""
Azhar EyeSight - Prediction & Inference Engine
Handles image input validation, model loading, classification inference,
confidence calculation, and optional Grad-CAM explainability.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn.functional as F

import config
from utils.preprocessing import preprocess_image_for_inference, safe_load_image
from utils.model import load_checkpoint, get_device
from utils.visualization import GradCAM, overlay_gradcam_on_image


class EyeDiseasePredictor:
    """
    Inference wrapper for the Azhar EyeSight ResNet18 classification model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = config.MODEL_PATH,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the predictor by loading the trained PyTorch weights.
        
        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If model fails to initialize.
        """
        self.model_path = Path(model_path)
        self.device = device if device is not None else get_device()
        self.model = None
        self.checkpoint_info = {}
        self.gradcam_engine = None

        self._load_model()

    def _load_model(self) -> None:
        """Loads and prepares the model for evaluation."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.model_path}'. "
                "Ensure training has finished via 'python train.py' or place weights at this path."
            )

        self.model, self.checkpoint_info = load_checkpoint(
            model_path=self.model_path,
            device=self.device,
            num_classes=config.NUM_CLASSES
        )
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        generate_gradcam: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a retinal fundus image.
        
        Args:
            image_input: File path, byte stream, or PIL Image.
            generate_gradcam: If True, computes Grad-CAM explainability overlay.
            
        Returns:
            Dictionary containing:
            - predicted_class: string identifier (e.g. 'cataract')
            - predicted_label: human-readable name (e.g. 'Cataract')
            - confidence: float probability (0.0 to 1.0)
            - probabilities: dict of class_name -> float probability
            - original_image: PIL.Image of verified input
            - gradcam_overlay: PIL.Image with Grad-CAM heatmap (or None)
            - gradcam_heatmap: 2D numpy array [224, 224] (or None)
            - description: medical overview of the predicted class
            - disclaimer: standard educational disclaimer
        """
        # Validate and preprocess image
        batch_tensor, pil_image = preprocess_image_for_inference(image_input)
        input_tensor = batch_tensor.to(self.device)

        # Run model inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs_tensor = F.softmax(logits, dim=1)[0]

        # Extract predictions
        probs = probs_tensor.cpu().numpy()
        pred_idx = int(torch.argmax(probs_tensor).item())
        pred_class = config.CLASSES[pred_idx]
        pred_label = config.CLASS_LABELS.get(pred_class, pred_class)
        confidence = float(probs[pred_idx])

        # Map all class probabilities
        class_probabilities = {
            cls: float(probs[i])
            for i, cls in enumerate(config.CLASSES)
        }

        # Optional Grad-CAM computation
        gradcam_overlay = None
        gradcam_heatmap = None
        if generate_gradcam:
            try:
                gc = GradCAM(self.model)
                gradcam_heatmap = gc.generate_heatmap(input_tensor, target_class_idx=pred_idx)
                gradcam_overlay = overlay_gradcam_on_image(pil_image, gradcam_heatmap)
                gc.remove_hooks()
            except Exception as e:
                # Log warning but do not fail the prediction
                print(f"[Warning] Grad-CAM generation failed: {e}", file=sys.stderr)

        return {
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": class_probabilities,
            "original_image": pil_image,
            "gradcam_overlay": gradcam_overlay,
            "gradcam_heatmap": gradcam_heatmap,
            "description": config.CLASS_DESCRIPTIONS.get(pred_class, ""),
            "device": str(self.device),
            "disclaimer": config.MEDICAL_DISCLAIMER
        }


def main():
    """Command-line interface for retinal image prediction."""
    parser = argparse.ArgumentParser(description="Azhar EyeSight - Retinal Disease Prediction CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus image (.jpg, .jpeg, .png)")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH), help="Path to trained PyTorch weights")
    parser.add_argument("--gradcam", action="store_true", help="Generate and display/save Grad-CAM heatmap overlay")
    parser.add_argument("--save_gradcam", type=str, default=None, help="Save path for Grad-CAM overlay image")

    args = parser.parse_args()

    try:
        predictor = EyeDiseasePredictor(model_path=args.model)
        result = predictor.predict(args.image, generate_gradcam=args.gradcam)

        print("\n" + "=" * 60)
        print(" AZHAR EYESIGHT – INFERENCE RESULT")
        print("=" * 60)
        print(f"Predicted Diagnosis : {result['predicted_label']} ({result['predicted_class']})")
        print(f"Confidence Score    : {result['confidence'] * 100:.2f}%")
        print("-" * 60)
        print("Class Probability Breakdown:")
        for cls, prob in result["probabilities"].items():
            label = config.CLASS_LABELS.get(cls, cls)
            bar = "#" * int(prob * 30)
            print(f"  {label:<22}: {prob * 100:6.2f}% | {bar}")
        print("-" * 60)
        print(f"Medical Overview    : {result['description']}")
        print("=" * 60)
        print(f"\n[DISCLAIMER] {result['disclaimer']}\n")

        if args.gradcam and result["gradcam_overlay"] is not None:
            if args.save_gradcam:
                out_path = Path(args.save_gradcam)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                result["gradcam_overlay"].save(out_path)
                print(f"[Grad-CAM] Attention heatmap saved to: {out_path.resolve()}")
            else:
                default_out = Path("results") / "latest_gradcam.png"
                default_out.parent.mkdir(parents=True, exist_ok=True)
                result["gradcam_overlay"].save(default_out)
                print(f"[Grad-CAM] Attention heatmap saved to: {default_out.resolve()}")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}\n", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n[Invalid Input] {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
