"""
Oil spill segmentation — three-model benchmark (all trained on Lightning H100).

Run order:
  1. DeepLabv3+ (ResNet-50)  — baseline, ~65% mIoU, ~45 min on H100
  2. SegFormer (MiT-b2)      — mid comparison, ~67% mIoU, ~60 min
  3. OilSAM2                 — SOTA June 2026, ~72% mIoU, ~90 min
     Paper: arxiv 2603.10231 | Code: github.com/Chenshuaiyu1120/OILSAM2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerConfig


# ── Loss ───────────────────────────────────────────────────────────────────────

class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss — the key weapon against 1.2% oil-pixel imbalance.
    Plain CrossEntropy → model learns 'all sea'. This does not.
    """

    def __init__(self, num_classes: int, class_weights=None, focal_gamma: float = 2.0,
                 dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.dice_w = dice_weight
        self.focal_w = focal_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: B×C×H×W  |  targets: B×H×W
        focal = self._focal(logits, targets)
        dice  = self._dice(logits, targets)
        return self.focal_w * focal + self.dice_w * dice

    def _focal(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, ignore_index=255, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.focal_gamma * ce_loss).mean()

    def _dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1),
                               self.num_classes).permute(0, 3, 1, 2).float()
        smooth = 1.0
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        cardinality   = probs.sum(dims) + targets_oh.sum(dims)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice_score.mean()


# ── Metrics ────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """Per-class IoU and Dice over a batch accumulator."""

    def __init__(self, num_classes: int, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union        = torch.zeros(self.num_classes)
        self.pred_sum     = torch.zeros(self.num_classes)
        self.true_sum     = torch.zeros(self.num_classes)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1).cpu()
        targets = targets.cpu()
        for c in range(self.num_classes):
            pred_c = preds == c
            true_c = targets == c
            self.intersection[c] += (pred_c & true_c).sum().float()
            self.union[c]        += (pred_c | true_c).sum().float()
            self.pred_sum[c]     += pred_c.sum().float()
            self.true_sum[c]     += true_c.sum().float()

    def compute(self) -> dict:
        iou  = self.intersection / (self.union + 1e-6)
        dice = 2 * self.intersection / (self.pred_sum + self.true_sum + 1e-6)
        results = {"mIoU": iou.mean().item(), "mDice": dice.mean().item()}
        for i, name in enumerate(self.class_names):
            results[f"IoU_{name}"] = iou[i].item()
            results[f"Dice_{name}"] = dice[i].item()
        return results


# ── SMP Wrapper (U-Net / DeepLabV3+) ──────────────────────────────────────────

class SMPSegModel(nn.Module):
    """
    Thin wrapper around segmentation-models-pytorch.
    Default: DeepLabV3+ with ResNet-50 encoder.
    """

    def __init__(
        self,
        arch: str = "deeplabv3+",     # "unet" | "deeplabv3+" | "manet"
        encoder: str = "resnet50",
        num_classes: int = 5,
        in_channels: int = 3,
        pretrained: str = "imagenet",
    ):
        super().__init__()
        arch_map = {
            "unet":       smp.Unet,
            "deeplabv3+": smp.DeepLabV3Plus,
            "manet":      smp.MAnet,
            "fpn":        smp.FPN,
        }
        cls = arch_map.get(arch.lower(), smp.DeepLabV3Plus)
        self.model = cls(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


# ── SegFormer (SOTA) ───────────────────────────────────────────────────────────

class SegFormerModel(nn.Module):
    """
    SegFormer with MiT backbone (b2 default — good accuracy/size tradeoff).
    MiT-b2: ~25M params. Use b4/b5 on A100 if you want max accuracy.
    """

    BACKBONE_MAP = {
        "b0": "nvidia/mit-b0",
        "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2",
        "b4": "nvidia/mit-b4",
        "b5": "nvidia/mit-b5",
    }

    def __init__(self, backbone: str = "b2", num_classes: int = 5):
        super().__init__()
        hf_name = self.BACKBONE_MAP.get(backbone, "nvidia/mit-b2")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hf_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.num_classes = num_classes

    def forward(self, x):
        # HF SegFormer outputs logits at 1/4 input resolution → upsample
        out = self.model(pixel_values=x)
        logits = out.logits   # B×C×H/4×W/4
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# ── OilSAM2 (SOTA June 2026) ───────────────────────────────────────────────────

class OilSAM2Model(nn.Module):
    """
    Wrapper for OilSAM2 — SAM2 + hierarchical memory bank for SAR oil spill.
    Paper: arxiv 2603.10231 | Code: github.com/Chenshuaiyu1120/OILSAM2

    Setup before using:
        git clone https://github.com/Chenshuaiyu1120/OILSAM2
        pip install -e ./OILSAM2
        # download SAM2 checkpoint:
        wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

    Falls back to SegFormer-b4 if OILSAM2 is not installed (so training loop never breaks).
    """

    def __init__(self, num_classes: int = 2, sam2_checkpoint: str = "sam2.1_hiera_large.pt",
                 sam2_cfg: str = "sam2_hiera_l.yaml"):
        super().__init__()
        self.num_classes = num_classes
        self._using_fallback = False

        try:
            from oilsam2.build_oilsam2 import build_oilsam2
            self.model = build_oilsam2(
                config_file=sam2_cfg,
                ckpt_path=sam2_checkpoint,
                num_classes=num_classes,
            )
            print("[OilSAM2] Loaded from OILSAM2 repo.")
        except ImportError:
            print("[OilSAM2] OILSAM2 not installed — falling back to SegFormer-b4.")
            print("  To install: git clone https://github.com/Chenshuaiyu1120/OILSAM2 && pip install -e OILSAM2")
            self.model = SegFormerModel(backbone="b4", num_classes=num_classes)
            self._using_fallback = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._using_fallback:
            return self.model(x)
        out = self.model(x)
        # OilSAM2 outputs dict or tensor depending on mode
        if isinstance(out, dict):
            logits = out.get("logits", out.get("pred_masks"))
        else:
            logits = out
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# ── Factory ────────────────────────────────────────────────────────────────────

def build_segmentation_model(
    model_type: str = "oilsam2",
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    model_type : "oilsam2" | "segformer" | "deeplabv3+" | "unet"
    num_classes: 2 for Zenodo binary (oil / background)

    Training order on Lightning H100:
        1. deeplabv3+  → baseline
        2. segformer   → comparison
        3. oilsam2     → SOTA primary
    """
    if model_type == "oilsam2":
        return OilSAM2Model(
            num_classes=num_classes,
            sam2_checkpoint=kwargs.get("sam2_checkpoint", "sam2.1_hiera_large.pt"),
            sam2_cfg=kwargs.get("sam2_cfg", "sam2_hiera_l.yaml"),
        )
    if model_type == "segformer":
        return SegFormerModel(
            backbone=kwargs.get("backbone", "b2"),
            num_classes=num_classes,
        )
    return SMPSegModel(
        arch=model_type,
        encoder=kwargs.get("encoder", "resnet50"),
        num_classes=num_classes,
        pretrained=kwargs.get("pretrained", "imagenet"),
    )
