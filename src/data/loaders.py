"""
Oil spill and vessel dataset loaders.
Handles:
  - Krestenitis/M4D 5-class (sea/oil/look-alike/ship/land)
  - Zenodo binary (oil=1 / background=0)  — raw dB GeoTIFF, needs dB→linear
  - SOS (Refined Deep-SAR, Zenodo 15298010) binary — grayscale PNG, already intensity
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
        cache_dir=None,              # if set, cache preprocessed bands here (float16 .npy)
    ):
        self.root = Path(root)
        self.split = split
        self.dataset_type = dataset_type
        self.transform = transform
        self.img_size = img_size
        self.val_split = val_split
        self.split_seed = split_seed
        self.num_classes = 5 if dataset_type == "krestenitis" else 2  # sos + zenodo = binary
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths, self.mask_paths = self._collect_paths()
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}"
        print(f"[OilSpillDataset] {dataset_type}/{split}: {len(self.image_paths)} samples, {self.num_classes} classes")

    def _collect_paths(self):
        if self.dataset_type == "krestenitis":
            return self._collect_krestenitis()
        if self.dataset_type == "sos":
            return self._collect_sos()
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
        # recursive + stem-matched: handles .tif nested in subfolders after .7z extraction,
        # and matches image<->mask by filename stem (robust to differing folder layouts)
        images = sorted(img_dir.rglob("*.tif")) + sorted(img_dir.rglob("*.tiff"))
        mask_by_stem = ({m.stem: m for m in mask_dir.rglob("*") if m.is_file()}
                        if mask_dir.exists() else {})
        pairs = [(im, mask_by_stem[im.stem]) for im in images if im.stem in mask_by_stem]

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

        img_paths = [im for im, _ in sel]
        msk_paths = [mk for _, mk in sel]
        # Track which images are hard negatives (p2l_ look-alikes OR p2n_ no-oil)
        # so the dataloader can upsample them.  Both contribute to false positives:
        # p2l_ visually mimics oil (ambiguous texture), p2n_ is pure background but
        # training on more negatives discourages the model from over-predicting oil.
        self.is_hard_negative = [
            p.stem.startswith("p2l_") or p.stem.startswith("p2n_")
            for p in img_paths
        ]
        return img_paths, msk_paths

    def _collect_sos(self):
        """
        Refined Deep-SAR SOS dataset (Zenodo 15298010).
        Zip structure: images/{train,val}/*.png + masks/{train,val}/*.png
        Two sources identified by filename prefix:
          palsar_*   → ALOS PALSAR L-band (Gulf of Mexico)
          sentinel_* → Sentinel-1A C-band (Persian Gulf)
        Grayscale PNG 256×256, binary masks (0=background, 255=oil).
        Skips macOS resource forks (._* files, .DS_Store).
        """
        img_base  = self.root / "images"
        mask_base = self.root / "masks"
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        def _is_real(p: Path) -> bool:
            return p.suffix.lower() in exts and not p.name.startswith("._") and p.name != ".DS_Store"

        # Use the baked-in train/val split from the zip structure.
        # Fall back to flat layout (legacy extraction) if subdirs don't exist.
        if self.split in ("test", "val"):
            subdirs = ["val"]
        elif self.split == "train":
            subdirs = ["train"]
        else:  # "all"
            subdirs = ["train", "val"]

        def _collect_dir(base: Path, sub: str):
            d = base / sub
            if d.exists():
                return sorted(p for p in d.iterdir() if _is_real(p))
            # fallback: flat layout (old extraction without subdir preservation)
            return sorted(p for p in base.iterdir() if _is_real(p))

        images = []
        for sd in subdirs:
            images += _collect_dir(img_base, sd)

        # Build mask lookup across all splits (stem → path)
        mask_by_stem = {}
        for sd in ["train", "val"]:
            d = mask_base / sd
            if d.exists():
                for m in d.iterdir():
                    if _is_real(m):
                        mask_by_stem[m.stem] = m
        if not mask_by_stem:  # flat fallback
            for m in mask_base.iterdir():
                if _is_real(m):
                    mask_by_stem[m.stem] = m

        pairs = [(im, mask_by_stem[im.stem]) for im in images if im.stem in mask_by_stem]
        self.is_hard_negative = [False] * len(pairs)
        return [im for im, _ in pairs], [mk for _, mk in pairs]

    def __len__(self):
        return len(self.image_paths)

    def _preprocess_bands(self, path: Path) -> np.ndarray:
        """C×H×W float32 [0,1] preprocessed bands, cached to disk if cache_dir is set.

        The dB→linear + Lee filter is expensive and identical every epoch, so the
        first epoch computes + caches it (float16); later epochs just np.load it.
        Augmentation still runs per-epoch in __getitem__, so variety is preserved.
        """
        if self.cache_dir is not None:
            cpath = self.cache_dir / f"{path.stem}.npy"
            if cpath.exists():
                return np.load(cpath).astype(np.float32)
        with rasterio.open(path) as src:
            raw = src.read().astype(np.float32)      # C×H×W (dB Sigma0 for Zenodo)
        # SAME pipeline as inference: per-band dB→linear (auto-detected), Lee speckle
        # filter, percentile-clip + normalize to [0,1]. No train/serve skew.
        proc = preprocess_sar_bands(raw)             # C×H×W float32 in [0,1]
        if self.cache_dir is not None:
            # atomic write (safe under many workers): unique tmp ending in .npy so
            # np.save doesn't append a second .npy, then rename into place.
            tmp = self.cache_dir / f"{path.stem}.{os.getpid()}.tmp.npy"
            np.save(tmp, proc.astype(np.float16))
            os.replace(tmp, cpath)
        return proc

    def _load_image(self, path: Path) -> np.ndarray:
        """Load SAR image as float32 H×W×3.

        Zenodo GeoTIFF:  2-band raw dB → dB→linear + Lee + normalize via sar_preprocess.
        SOS / other PNG: grayscale already-processed intensity → /255 → replicate to 3 ch.
        """
        if path.suffix.lower() in {".tif", ".tiff"}:
            proc = self._preprocess_bands(path)      # C×H×W float32 [0,1] (cached)
            img = np.transpose(proc, (1, 2, 0))      # H×W×C
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:
                # Oil signal lives in band 2 (~9 dB oil-vs-water) vs band 1 (~1 dB).
                # Duplicate the strong band: [VV, VH, VH] instead of [VV, VH, VV].
                img = np.concatenate([img, img[:, :, 1:2]], axis=2)
        else:
            # SOS: grayscale PNG (single channel intensity, 0–255, already preprocessed)
            # Convert to RGB by replicating — encoder expects 3 channels.
            pil = Image.open(path)
            if pil.mode != "RGB":
                pil = pil.convert("L")  # ensure grayscale, then replicate
                arr = np.array(pil, dtype=np.float32) / 255.0   # H×W [0,1]
                img = np.stack([arr, arr, arr], axis=2)          # H×W×3
            else:
                img = np.array(pil, dtype=np.float32) / 255.0   # H×W×3
        return img.astype(np.float32)  # H×W×3, float32

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask as H×W int64 class indices."""
        if self.dataset_type == "zenodo":
            with rasterio.open(path) as src:
                mask = src.read(1).astype(np.int64)
        elif self.dataset_type == "sos":
            # SOS masks are grayscale PNG: 0=background, 255=oil (or 1=oil in some versions).
            arr = np.array(Image.open(path).convert("L"), dtype=np.int64)
            mask = (arr > 0).astype(np.int64)   # normalise any non-zero value → class 1
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
            # albumentations ≥1.4 uses size=(h,w) instead of height= width=
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0)),
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
    cache_preproc: bool = True,
    upsample_lookalike: float = 1.0,
    sos_root: str = None,
):
    """
    upsample_lookalike : weight multiplier for p2l_+p2n_* hard-negative training samples.
      Default 1.0 = uniform. 2.0 = hard negatives appear ~2x per epoch.
      Only active for dataset_type=zenodo where stem prefixes identify source.

    sos_root : if set, load SOS dataset and concatenate with Zenodo (train-only mode).
      SOS has its own train/val split (images/train/ + images/val/).
      SOS train → concat with Zenodo train.
      SOS val   → concat with Zenodo val (broader validation signal).
      SOS images are grayscale PNG (already intensity), no dB->linear applied.
    """
    # Cache the expensive dB->linear+Lee preprocessing once (shared across all models
    # and epochs) so the GPU isn't starved re-filtering the same images every epoch.
    cache_dir = Path(root) / ".preproc_cache" if cache_preproc else None
    train_ds = OilSpillDataset(
        root, split="train", dataset_type=dataset_type,
        transform=get_oil_transforms(img_size, "train"), img_size=img_size,
        cache_dir=cache_dir,
    )
    val_ds = OilSpillDataset(
        root, split="test", dataset_type=dataset_type,
        transform=get_oil_transforms(img_size, "val"), img_size=img_size,
        cache_dir=cache_dir,
    )
    # Training is data-bound (per-image dB->linear + Lee filter at native res on CPU),
    # so on a fast GPU (H100) keep workers warm and prefetch deeper to hide that latency.
    extra = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}

    # Optionally mix in SOS dataset (cross-domain: PALSAR + Sentinel-1A from different regions)
    # SOS has its own train/val baked-in split — use both halves.
    if sos_root is not None and dataset_type == "zenodo":
        from torch.utils.data import ConcatDataset
        sos_train = OilSpillDataset(
            sos_root, split="train", dataset_type="sos",
            transform=get_oil_transforms(img_size, "train"), img_size=img_size,
        )
        sos_val = OilSpillDataset(
            sos_root, split="val", dataset_type="sos",
            transform=get_oil_transforms(img_size, "val"), img_size=img_size,
        )
        # Merge SOS train into Zenodo train
        n_sos_tr = len(sos_train)
        train_ds = ConcatDataset([train_ds, sos_train])
        if hasattr(train_ds.datasets[0], "is_hard_negative"):
            train_ds.is_hard_negative = (train_ds.datasets[0].is_hard_negative
                                         + [False] * n_sos_tr)
        # Merge SOS val into Zenodo val
        val_ds = ConcatDataset([val_ds, sos_val])
        print(f"[DataLoader] SOS mixed in — "
              f"train: Zenodo({len(train_ds.datasets[0])}) + SOS({n_sos_tr}) = {len(train_ds)} | "
              f"val: Zenodo({len(val_ds.datasets[0])}) + SOS({len(sos_val)}) = {len(val_ds)}")

    # Look-alike upsampling: give p2l_*+p2n_* stems a higher sampling weight so the
    # model sees more FP-inducing examples per epoch.  Uses WeightedRandomSampler
    # (with replacement) instead of shuffle=True.
    train_shuffle = True
    train_sampler = None
    if (dataset_type == "zenodo" and upsample_lookalike > 1.0
            and hasattr(train_ds, "is_hard_negative")
            and train_ds.is_hard_negative):
        import torch
        from torch.utils.data import WeightedRandomSampler
        weights = torch.tensor(
            [upsample_lookalike if hn else 1.0 for hn in train_ds.is_hard_negative],
            dtype=torch.float,
        )
        n_hn = sum(train_ds.is_hard_negative)
        print(f"[DataLoader] hard-negative upsampling ×{upsample_lookalike:.1f} "
              f"({n_hn}/{len(train_ds)} p2l_+p2n_ samples in train split)")
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=train_shuffle, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True, **extra)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, **extra)
    return train_loader, val_loader


# ── Compute class weights for imbalanced oil data ──────────────────────────────

def compute_class_weights(dataset: OilSpillDataset, num_classes: int) -> torch.Tensor:
    """
    Count pixels per class across the training set → inverse-frequency weights.
    Oil pixels ≈ 1.2% — without this, the model predicts 'all sea'.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    print("Computing class weights (one-time scan)...")
    # Read MASKS ONLY. Iterating the dataset would call __getitem__, which fully
    # preprocesses each image (dB->linear + Lee filter at 2048^2) and then discards
    # it here — turning a quick pixel-count into a multi-minute scan (×N models).
    mask_paths = getattr(dataset, "mask_paths", None)
    if mask_paths is not None:
        for mp in mask_paths:
            mask_np = dataset._load_mask(mp)
            for c in range(num_classes):
                counts[c] += (mask_np == c).sum()
    else:  # fallback: dataset without exposed mask_paths
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
