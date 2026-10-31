"""
gradcam.py
----------
Grad-CAM for DualAttentionResNet50, hooked at model.head_conv (i.e. the
attended, post-dual-attention feature map that directly feeds the
classification head). This lets the heatmap reflect the actual regions
driving the prediction, downstream of both attention mechanisms.

Usage:
    from gradcam import GradCAM
    cam = GradCAM(model)
    heatmap, pred_class = cam.generate(input_tensor)   # input_tensor: 1x3x224x224
    overlay = cam.overlay_on_image(heatmap, original_pil_image)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.get_gradcam_target_layer()

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: 1 x 3 x 224 x 224, normalized, on the same device as model.
        class_idx: which class to explain; defaults to the predicted class.
        Returns: (heatmap [224,224] in [0,1], predicted_class_idx, probs [5])
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # 1 x C x 1 x 1
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # 1 x 1 x h x w
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx, probs.detach().cpu().numpy()

    @staticmethod
    def overlay_on_image(heatmap, pil_image, alpha=0.45):
        """heatmap: 224x224 in [0,1]; pil_image: PIL RGB image (any size, will be resized to 224x224)."""
        pil_image = pil_image.convert("RGB").resize((224, 224))
        base = np.array(pil_image)

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        overlay = (colored * alpha + base * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(overlay)
