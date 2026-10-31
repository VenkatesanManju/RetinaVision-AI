"""
app.py
------
RetinaVision AI - Flask backend.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Serves a plain HTML/CSS/JS frontend (templates/index.html + static/) and
exposes a single JSON prediction endpoint that:
    1. Accepts an uploaded fundus image
    2. Runs it through the trained DualAttentionResNet50
    3. Runs Grad-CAM on the predicted class
    4. Returns the predicted stage, confidence, full class-probability
       breakdown, and a base64-encoded Grad-CAM heatmap overlay image.

No Streamlit / Gradio — pure Flask + static HTML/CSS/JS as requested.
"""

import os
import io
import base64
import random
import sys
import json

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import build_ensemble_model
from dataset import get_eval_transforms, CLASS_NAMES
from gradcam import GradCAM

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RECOVERED_ACCURACY_CHECKPOINT = os.path.join(APP_DIR, "checkpoints", "best_model_accuracy_RECOVERED.pth")
CHECKPOINTS = {
    "accuracy": RECOVERED_ACCURACY_CHECKPOINT,
    "accuracy_recovered": RECOVERED_ACCURACY_CHECKPOINT,
    "kappa": os.path.join(APP_DIR, "checkpoints", "best_model_kappa.pth"),
}
DEFAULT_CHECKPOINT = os.environ.get("RETINAVISION_CHECKPOINT", "accuracy")

DISPLAY_NAMES = {
    "No_DR": "0 - No DR",
    "Mild": "1 - Mild DR",
    "Moderate": "2 - Moderate DR",
    "Severe": "3 - Severe DR",
    "Proliferate_DR": "4 - Proliferative DR",
}

STAGE_DESCRIPTIONS = {
    "No_DR": "No visible signs of diabetic retinopathy detected.",
    "Mild": "Early stage: microaneurysms present. Routine monitoring advised.",
    "Moderate": "Blood vessel changes present. Follow-up with a specialist recommended.",
    "Severe": "Advanced stage: severe hemorrhages or venous beading present. Immediate specialist care needed.",
    "Proliferate_DR": "Most advanced stage: abnormal new vessel growth. Urgent ophthalmology referral required.",
}

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
cam = None
loaded_checkpoint_name = None
eval_transform = get_eval_transforms()
model_accuracy = None
model_weighted_kappa = None
metrics_source = None
branch_accuracies = {}


def load_model(checkpoint_name=DEFAULT_CHECKPOINT):
    global model, cam, model_accuracy, model_weighted_kappa, metrics_source
    global branch_accuracies, loaded_checkpoint_name

    if checkpoint_name not in CHECKPOINTS:
        raise ValueError(f"Unsupported checkpoint name '{checkpoint_name}'. Available keys: {sorted(CHECKPOINTS.keys())}")

    checkpoint_path = CHECKPOINTS[checkpoint_name]
    print(f"Requested checkpoint: {checkpoint_name} -> {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}. "
            "The app cannot serve an untrained model. Please ensure the verified checkpoint exists."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "legacy_resnet50")
    feature_layer = ckpt.get("feature_layer", "layer3")
    resnet_weight = ckpt.get("resnet_weight", 0.7)

    if model_type != "resnet50_efficientnet_b0_ensemble":
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}: model_type={model_type}. "
            "This app requires the ensemble checkpoint trained with the current project."
        )

    model = build_ensemble_model(
        num_classes=5,
        feature_layer=feature_layer,
        pretrained=False,
        resnet_weight=resnet_weight,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = ckpt.get("val_metrics", {})
    model_accuracy = val_metrics.get("accuracy")
    model_weighted_kappa = val_metrics.get("weighted_kappa")
    metrics_source = "checkpoint validation metrics"

    branch_accuracies = {
        "resnet50": val_metrics.get("resnet_accuracy"),
        "efficientnet_b0": val_metrics.get("efficientnet_accuracy"),
        "ensemble": val_metrics.get("accuracy"),
    }

    epoch = ckpt.get("epoch", "unknown")
    selection_metric = ckpt.get("selection_metric", "unknown")
    print(
        "Loaded checkpoint: "
        f"path={checkpoint_path}, epoch={epoch}, selection_metric={selection_metric}, "
        f"accuracy={model_accuracy}, weighted_kappa={model_weighted_kappa}, "
        f"feature_layer={feature_layer}, resnet_weight={resnet_weight}"
    )

    model.eval()
    cam = GradCAM(model)
    loaded_checkpoint_name = checkpoint_name


def pil_to_base64(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device),
                     "checkpoint_loaded": loaded_checkpoint_name in CHECKPOINTS and os.path.exists(CHECKPOINTS[loaded_checkpoint_name]),
                     "checkpoint_name": loaded_checkpoint_name,
                     "available_checkpoints": {name: os.path.exists(path) for name, path in CHECKPOINTS.items()},
                     "model_accuracy": model_accuracy,
                     "model_weighted_kappa": model_weighted_kappa,
                     "metrics_source": metrics_source,
                     "branch_accuracies": branch_accuracies})


@app.route("/sample/<cls>")
def sample(cls):
    """Return a random real held-out test image (base64 PNG) for the given class,
    so the frontend's Quick Test Sample buttons work without the user hunting
    through folders manually."""
    if cls not in CLASS_NAMES:
        return jsonify({"error": f"Unknown class '{cls}'"}), 400

    candidates = [
        os.path.join(APP_DIR, "data", "split", "test", cls),
        os.path.join(APP_DIR, "Test", cls),
        os.path.join(APP_DIR, "test", cls),
    ]
    test_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            test_dir = cand
            break

    if test_dir is None:
        # Try case-insensitive matching in Test or test folder
        for root_folder in [os.path.join(APP_DIR, "Test"), os.path.join(APP_DIR, "test")]:
            if os.path.isdir(root_folder):
                for item in os.listdir(root_folder):
                    if item.lower() == cls.lower() and os.path.isdir(os.path.join(root_folder, item)):
                        test_dir = os.path.join(root_folder, item)
                        break

    if test_dir is None or not os.path.isdir(test_dir):
        return jsonify({"error": f"No test images found for '{cls}'."}), 404

    files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return jsonify({"error": f"No image files inside {test_dir}"}), 404

    chosen = random.choice(files)
    pil_img = Image.open(os.path.join(test_dir, chosen)).convert("RGB")
    return jsonify({
        "filename": chosen,
        "class": cls,
        "image_b64": pil_to_base64(pil_img),
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model
    requested_checkpoint = request.form.get("checkpoint", DEFAULT_CHECKPOINT)
    if model is None or requested_checkpoint != loaded_checkpoint_name:
        load_model(requested_checkpoint)

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Field name must be 'image'."}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    input_tensor = eval_transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        ensemble_log_probs = model(input_tensor.to(device))
        probs = ensemble_log_probs.exp()[0].cpu().numpy()
    pred_idx = int(probs.argmax())

    # Grad-CAM needs gradients, so run it separately (requires_grad path)
    heatmap, cam_pred_idx, cam_probs = cam.generate(input_tensor.clone(), class_idx=pred_idx)
    overlay = cam.overlay_on_image(heatmap, pil_img)

    pred_class_name = CLASS_NAMES[pred_idx]

    class_probs = [
        {
            "class_index": i,
            "label": DISPLAY_NAMES[CLASS_NAMES[i]],
            "probability": round(float(probs[i]) * 100, 1),
        }
        for i in range(len(CLASS_NAMES))
    ]

    model_confidence = round(float(probs[pred_idx]) * 100, 1)
    response = {
        "predicted_class_index": pred_idx,
        "predicted_label": DISPLAY_NAMES[pred_class_name],
        "description": STAGE_DESCRIPTIONS[pred_class_name],
        "confidence": model_confidence,
        "model_accuracy": round(float(model_accuracy) * 100, 1) if model_accuracy is not None else None,
        "model_weighted_kappa": round(float(model_weighted_kappa), 4) if model_weighted_kappa is not None else None,
        "metrics_source": metrics_source,
        "checkpoint_name": loaded_checkpoint_name,
        "branch_accuracies": branch_accuracies,
        "ensemble": "ResNet50 + EfficientNet-B0",
        "class_probabilities": class_probs,
        "input_image_b64": pil_to_base64(pil_img.resize((224, 224))),
        "gradcam_image_b64": pil_to_base64(overlay),
        "checkpoint_loaded": loaded_checkpoint_name in CHECKPOINTS and os.path.exists(CHECKPOINTS[loaded_checkpoint_name]),
    }
    return jsonify(response)


if __name__ == "__main__":
    load_model(DEFAULT_CHECKPOINT)
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    load_model(DEFAULT_CHECKPOINT)
