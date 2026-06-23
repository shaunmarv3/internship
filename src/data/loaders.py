"""
Oil spill and vessel dataset loaders.
Handles both:
  - Krestenitis/M4D 5-class (sea/oil/look-alike/ship/land)
  - Zenodo binary (oil=1 / background=0)
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio

# Reuse the EXACT inference preprocessing so training matches live/GEE data.
# Zenodo oil images are Sigma0 in dB → this converts dB→linear, Lee-filters and
# percentile-normalizes, identical to what sar_preprocess does at inference time.
from .sar_preprocess import preprocess_sar_bands


# ── Class definitions ──────────────────────────────────────────────────────────

OIL_CLASSES_5 = {
    0: "sea",
    1: "oil_spill",
    2: "look_alike",
    3: "ship",
    4: "land",
}

OIL_CLASSES_BINARY = {
    0: "background",
    1: "oil_spill",
}

# Krestenitis color-coded mask → class index
# Masks are stored as RGB color images
COLOR_TO_CLASS = {
    (0,   0,   0): 0,   # sea       → black
    (0, 128, 128): 1,   # oil spill → teal/cyan
    (128, 0,   0): 2,   # look-alike→ red/maroon
    (128, 64,   0): 3,  # ship      → brown
    (0, 128,   0): 4,   # land      → green
}


def color_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an H×W×3 color mask to an H×W integer class map."""
    h, w = mask_rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.int64)
    for color, cls in COLOR_TO_CLASS.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        out[match] = cls
    return out


# ── Oil spill dataset ──────────────────────────────────────────────────────────

class OilSpillDataset(Dataset):
    """
    Works with both Krestenitis (5-class) and Zenodo (binary) layouts:

    Krestenitis layout:
        root/
          train/images/*.jpg   (or .png)
          train/labels/masks/*.png   (color-coded RGB)
          test/images/
          test/labels/masks/

    Zenodo layout:
        root/
          images/*.tif         (2048×2048, 2-channel or 1-channel)
          masks/*.tif          (2048×2048, binary uint8)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",        # "train" | "test" | "all"
        dataset_type: str = "krestenitis",  # "krestenitis" | "zenodo"
        transform=None,
        img_size: int = 512,
        val_split: float = 0.15,     # zenodo: fraction held out for validation
        split_seed: int = 42,        # zenodo: deterministic train/val partition
    ):
        self.root = Path(root)
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        self.img_size = img_size
        self.val_split = val_split
        self.split_seed = split_seed
        self.num_classes = 5 if dataset_type == "krestenitis" else 2

        self.image_paths, self.mask_paths = self._collect_paths()
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}"
        print(f"[OilSpillDataset] {dataset_type}/{split}: {len(self.image_paths)} samples, {self.num_classes} classes")

    def _collect_paths(self):
        if self.dataset_type == "krestenitis":
            return self._collect_krestenitis()
        return self._collect_zenodo()

    def _collect_krestenitis(self):
        splits = ["train", "test"] if self.split == "all" else [self.split]
        images, masks = [], []
        for sp in splits:
            img_dir = self.root / sp / "images"
            mask_dir = self.root / sp / "labels" / "masks"
            if not img_dir.exists():
                # try flat structure: root/images, root/masks
                img_dir = self.root / "images"
                mask_dir = self.root / "masks"
            for p in sorted(img_dir.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    stem = p.stem
                    # try matching mask by stem
                    mask = mask_dir / (stem + ".png")
                    if not mask.exists():
                        mask = mask_dir / (stem + "_mask.png")
                    if mask.exists():
                        images.append(p)
                        masks.append(mask)
        return images, masks

    def _collect_zenodo(self):
        """
        Zenodo has a single images/ + masks/ folder (no train/test subdirs).
        We carve a DETERMINISTIC, DISJOINT train/val partition here so that
        split="train" and split="test" never share images — otherwise the
        validation IoU is just training accuracy (data leakage).
        """
        img_dir = self.root / "images"
        mask_dir = self.root / "masks"
        images = sorted(img_dir.glob("*.tif"))
        pairs = [(p, mask_dir / p.name) for p in images]
        pairs = [(im, mk) for im, mk in pairs if mk.exists()]

        if self.split == "all" or self.val_split <= 0:
            sel = pairs
        else:
            rng = np.random.default_rng(self.split_seed)
            order = rng.permutation(len(pairs))
            n_val = max(1, int(round(len(pairs) * self.val_split)))
            val_idx = set(order[:n_val].tolist())
            if self.split in ("test", "val"):
                sel = [pairs[i] for i in range(len(pairs)) if i in val_idx]
            else:  # "train"
                sel = [pairs[i] for i in range(len(pairs)) if i not in val_idx]

        return [im for im, _ in sel], [mk for _, mk in sel]

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: Path) -> np.ndarray:
        """Load SAR image as float32 H×W×C (1 or 2 channels → replicate to 3 for encoders)."""
        if path.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(path) as src:
                raw = src.read().astype(np.float32)  # C×H×W (dB Sigma0 for Zenodo)
            # SAME pipeline as inference: per-band dB→linear (auto-detected), Lee speckle
            # filter, percentile-clip + normalize to [0,1]. No train/serve skew.
            proc = preprocess_sar_bands(raw)         # C×H×W float32 in [0,1]
            img = np.transpose(proc, (1, 2, 0))      # H×W×C
            # replicate to 3 channels if needed
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:
                img = np.concatenate([img, img[:, :, :1]], axis=2)
        else:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return img.astype(np.float32)  # H×W×3, float32

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as H×W int64 class indices."""
        if self.dataset_type == "zenodo":
            with rasterio.open(path) as src:
                mask = src.read(1).astype(np.int64)
        else:
            mask_img = np.array(Image.open(path).convert("RGB"))
            mask = color_mask_to_index(mask_img)
        return mask

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            # albumentations resize casts masks to int32 — CrossEntropy/one_hot need int64
            mask = augmented["mask"].long()
        else:
            # default resize + to tensor
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()

        return img, mask


def get_oil_transforms(img_size: int = 512, split: str = "train"):
    # NOTE: _load_image() already returns bands in [0, 1] (via sar_preprocess:
    # dB->linear, Lee, percentile-normalize). Therefore Normalize must use
    # max_pixel_value=1.0 (NOT the default 255.0), and GaussNoise variance must be
    # on a [0, 1] scale — otherwise contrast collapses (~58x) and noise swamps signal.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(0.0005, 0.005), p=0.3),   # SAR speckle ([0,1] scale)
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
        ToTensorV2(),
    ])


def get_oil_dataloaders(
    root: str,
    dataset_type: str = "krestenitis",
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
):
    train_ds = OilSpillDataset(
        root, split="train", dataset_type=dataset_type,
        transform=get_oil_transforms(img_size, "train"), img_size=img_size,
    )
    val_ds = OilSpillDataset(
        root, split="test", dataset_type=dataset_type,
        transform=get_oil_transforms(img_size, "val"), img_size=img_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Compute class weights for imbalanced oil data ──────────────────────────────

def compute_class_weights(dataset: OilSpillDataset, num_classes: int) -> torch.Tensor:
    """
    Count pixels per class across the training set → inverse-frequency weights.
    Oil pixels ≈ 1.2% — without this, the model predicts 'all sea'.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    print("Computing class weights (one-time scan)...")
    for _, mask in dataset:
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        for c in range(num_classes):
            counts[c] += (mask_np == c).sum()
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    names = list(OIL_CLASSES_5.values()) if num_classes == 5 else list(OIL_CLASSES_BINARY.values())
    print(f"Class counts: {dict(zip(names, counts.astype(int)))}")
    print(f"Class weights: {weights.round(3)}")
    return torch.tensor(weights, dtype=torch.float32)
