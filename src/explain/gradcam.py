"""
Grad-CAM saliency maps for SAR segmentation models.
Shows WHICH pixels drove the "oil" or "vessel" decision → Module 5 explainability.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_target_layer(model, model_type: str = "segformer"):
    """Return the last conv/attention layer suitable for Grad-CAM.

    For SegFormer the best target is decode_head.classifier — it is a Conv2d that
    maps 256 fused decoder features -> num_classes, so its INPUT activations are
    4D (B×256×H/4×W/4). Targeting encoder LayerNorm gives 3D token sequences
    which confuse pytorch-grad-cam (needs a reshape_transform). Decoder conv
    gives spatial maps directly, no reshape needed.
    """
    if model_type == "segformer":
        # prefer decode_head.classifier (4D spatial — most reliable)
        hf_model = getattr(model, "model", model)
        dh = getattr(hf_model, "decode_head", None)
        if dh is not None:
            clf = getattr(dh, "classifier", None) or getattr(dh, "linear_pred", None)
            if clf is not None:
                return [clf]
        # fallback: encoder's last LayerNorm (3D tokens — may not render perfectly)
        seg = getattr(hf_model, "segformer", hf_model)
        if hasattr(seg, "stages"):
            return [seg.stages[-1].blocks[-1].layernorm_after]
        if hasattr(seg, "encoder") and hasattr(seg.encoder, "block"):
            return [seg.encoder.block[-1][-1].layer_norm_1]
    elif model_type in ("unet", "deeplabv3+"):
        return [model.model.decoder.blocks[-1]]
    return [list(model.modules())[-3]]


def segmentation_target(class_idx: int):
    """Grad-CAM target: average activation of the target class in the output."""
    class SegTarget:
        def __init__(self, ci): self.ci = ci
        def __call__(self, output):
            # pytorch-grad-cam >= 1.5 iterates over the batch, passing one
            # per-sample tensor (C×H×W, 3D) rather than the full batch (4D).
            if output.dim() == 4:
                return output[:, self.ci, :, :].mean()
            if output.dim() == 3:
                return output[self.ci].mean()   # per-sample C×H×W
            return output.mean()
    return [SegTarget(class_idx)]


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,    # 1×3×H×W, normalised
    image_rgb: np.ndarray,         # H×W×3, float [0,1] for overlay
    target_class: int = 1,         # 1 = oil_spill
    model_type: str = "segformer",
    save_path: str = None,
) -> np.ndarray:
    """
    Returns the Grad-CAM heatmap overlaid on the SAR image (H×W×3, uint8).
    target_class: 1=oil_spill, 2=look_alike, 3=ship, 4=land (Krestenitis)
    """
    target_layers = get_target_layer(model, model_type)
    # pytorch-grad-cam >= 1.5 removed the `use_cuda` arg — device is inferred from the
    # model. Ensure the model and input are on the same device before constructing.
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = segmentation_target(target_class)

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets,
    )[0]  # H×W

    image_rgb_f32 = image_rgb.astype(np.float32)
    if image_rgb_f32.max() > 1.0:
        image_rgb_f32 /= 255.0

    visualization = show_cam_on_image(image_rgb_f32, grayscale_cam, use_rgb=True)

    if save_path:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb_f32)
        plt.title("SAR Input")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM (class {target_class})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return visualization
