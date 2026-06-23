#!/usr/bin/env python
"""
Colab smoke test — exercise the WHOLE training stack on tiny synthetic data before
spending GPU hours on Lightning.ai. It runs the REAL training CLIs via subprocess
(same entry points Lightning will use), so import bugs, arg bugs, path bugs, W&B and
HF wiring all surface here first.

Run from the repo root (after `pip install -r requirements.txt`):

    python smoke_test.py                 # all stages
    python smoke_test.py --stage oil     # just one stage
    python smoke_test.py --skip detect   # skip the heavy YOLO download/run

Optional credential checks (only run if the env var is present):
    HF_TOKEN=hf_xxx   python smoke_test.py --stage hf       # real tiny push to smoke/ subfolder
                       python smoke_test.py --stage wandb     # W&B in OFFLINE mode (no login)

Everything lands under ./smoke_data/ and ./checkpoints are not touched (uses smoke dirs).
Exit code is non-zero if any stage fails, so it works in CI too.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# keep prints safe on non-UTF8 consoles (Windows cp1252); Colab/Linux are UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SD = ROOT / "smoke_data"
ALL_STAGES = ["imports", "preprocess", "oil", "detect", "fishing", "wandb", "hf"]

RESULTS = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def banner(msg):
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70, flush=True)


def run(cmd, env=None, timeout=1800):
    """Run a subprocess from repo root, stream output, return True on exit 0."""
    print(f"$ {PY} " + " ".join(cmd), flush=True)
    full_env = {**os.environ, **(env or {})}
    try:
        r = subprocess.run([PY] + cmd, cwd=str(ROOT), env=full_env, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"!! TIMEOUT after {timeout}s", flush=True)
        return False


# ── synthetic data ───────────────────────────────────────────────────────────

def make_oil_data(n=16, size=256):
    """Zenodo-style: data/oil/images/*.tif (2-band float) + masks/*.tif (uint8 0/1)."""
    import rasterio
    from rasterio.transform import from_origin

    img_dir = SD / "oil" / "images"
    msk_dir = SD / "oil" / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tr = from_origin(14.0, 37.5, 0.001, 0.001)

    for i in range(n):
        name = f"scene_{i:03d}.tif"
        img = (rng.random((2, size, size)).astype("float32") * 0.3)
        # plant a darker "oil" patch and a matching mask
        y0, x0 = rng.integers(20, size - 80, 2)
        img[:, y0:y0 + 60, x0:x0 + 60] *= 0.2
        mask = np.zeros((size, size), dtype="uint8")
        mask[y0:y0 + 60, x0:x0 + 60] = 1

        with rasterio.open(img_dir / name, "w", driver="GTiff", height=size, width=size,
                           count=2, dtype="float32", crs="EPSG:4326", transform=tr) as d:
            d.write(img)
        with rasterio.open(msk_dir / name, "w", driver="GTiff", height=size, width=size,
                           count=1, dtype="uint8", crs="EPSG:4326", transform=tr) as d:
            d.write(mask, 1)
    return str(SD / "oil")


def make_yolo_data(n_train=10, n_val=4, size=64):
    """HRSID-style YOLO: images/{train,val}, labels/{train,val}, data.yaml (1 class: ship)."""
    import cv2
    base = SD / "vessels"
    rng = np.random.default_rng(1)
    for split, n in (("train", n_train), ("val", n_val)):
        idir = base / "images" / split
        ldir = base / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = (rng.random((size, size, 3)) * 60).astype("uint8")
            lines = []
            if i % 4 != 0:                      # ~75% have a "ship", rest are negatives
                cx, cy = rng.uniform(0.3, 0.7, 2)
                cv2.rectangle(img, (int((cx - .1) * size), int((cy - .1) * size)),
                              (int((cx + .1) * size), int((cy + .1) * size)), (255, 255, 255), -1)
                lines.append(f"0 {cx:.4f} {cy:.4f} 0.2 0.2")
            cv2.imwrite(str(idir / f"img_{i:03d}.png"), img)
            (ldir / f"img_{i:03d}.txt").write_text("\n".join(lines))

    data_yaml = base / "data.yaml"
    data_yaml.write_text(
        f"path: {base.as_posix()}\n"
        f"train: images/train\nval: images/val\n"
        f"names:\n  0: ship\n"
    )
    return str(data_yaml)


def make_fishing_csv(n=240):
    """GFW-style fishing events CSV with the exact columns load_gfw_fishing_features uses."""
    import pandas as pd
    rng = np.random.default_rng(2)
    risk = rng.random(n) < 0.35                  # ~35% positive → both classes, stratifiable
    df = pd.DataFrame({
        "fishing.averageSpeedKnots":          np.where(risk, rng.uniform(1, 4, n), rng.uniform(8, 14, n)),
        "fishing.totalDistanceKm":            rng.uniform(5, 400, n),
        "fishing.averageDurationHours":       rng.uniform(1, 40, n),
        "distances.startDistanceFromPortKm":  np.where(risk, rng.uniform(80, 300, n), rng.uniform(1, 50, n)),
        "distances.startDistanceFromShoreKm": rng.uniform(1, 200, n),
        "distances.endDistanceFromPortKm":    rng.uniform(1, 300, n),
        "distances.endDistanceFromShoreKm":   rng.uniform(1, 200, n),
        "regions.mpa":      [(["mpa1"] if r and rng.random() < .6 else []) for r in risk],
        "regions.highSeas": [(["hs"] if rng.random() < .3 else []) for _ in range(n)],
        "regions.rfmo":     [[] for _ in range(n)],
        "fishing.vesselPublicAuthorizationStatus":
            np.where(risk, rng.choice(["unmatched", "partially_matched"], n),
                     "publicly_authorized"),
        "vessel.flag":           rng.choice(["ESP", "ITA", "CHN", "RUS", "FRA"], n),
        "fishing.potentialRisk": risk,
    })
    out = SD / "gfw" / "fishing_events_smoke.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


# ── stages ───────────────────────────────────────────────────────────────────

def stage_imports():
    banner("STAGE imports — every src module imports cleanly")
    mods = [
        "src.data.loaders", "src.data.sar_preprocess",
        "src.models.segmentation", "src.models.detection",
        "src.models.drift_prediction", "src.models.hf_utils",
        "src.fusion.ais_matching", "src.fusion.risk_scoring",
        "src.explain.gradcam", "src.explain.shap_explain", "src.viz.maps",
    ]
    code = "import importlib; " + "; ".join(f"importlib.import_module('{m}')" for m in mods) \
        + "; print('all imports OK')"
    return run(["-c", code])


def stage_preprocess():
    banner("STAGE preprocess — sar_preprocess on a synthetic dB scene")
    make_oil_data(n=1, size=256)
    tif = next((SD / "oil" / "images").glob("*.tif"))
    code = (
        "from src.data.sar_preprocess import preprocess_sar_tif; "
        f"c = preprocess_sar_tif(r'{tif}', out_dir=r'{SD/'pp_chips'}', chip_size=128, overlap=16); "
        "print('chips:', len(c)); assert len(c) >= 1"
    )
    return run(["-c", code])


def stage_oil():
    banner("STAGE oil (M3) — train_segmentation, deeplabv3+, 1 epoch, tiny")
    root = make_oil_data(n=16, size=256)
    ck = SD / "ck_oil"
    ok = run([
        "src/models/train_segmentation.py", "--model", "deeplabv3+",
        "--dataset_type", "zenodo", "--data_root", root,
        "--epochs", "1", "--batch_size", "2", "--img_size", "128",
        "--num_workers", "0", "--checkpoint_dir", str(ck), "--no_wandb",
    ])
    saved = list(ck.glob("best_*.pt"))
    print("oil checkpoints:", [p.name for p in saved])
    return ok and len(saved) >= 1


def stage_detect():
    banner("STAGE detect (M1) — train_detection, yolov8m, 1 epoch, imgsz=64 (downloads yolov8m.pt)")
    data_yaml = make_yolo_data()
    proj = SD / "ck_vessel"
    ok = run([
        "src/models/train_detection.py", "--model", "yolov8m",
        "--data", data_yaml, "--project", str(proj),
        "--epochs", "1", "--imgsz", "64", "--batch", "2", "--no_wandb",
    ])
    best = list(proj.glob("**/weights/best.pt"))
    print("vessel best.pt:", [str(p.relative_to(proj)) for p in best])
    return ok and len(best) >= 1


def stage_fishing(extra_env=None, wandb=False):
    label = "fishing (M2) — XGBoost + SHAP" + ("  [W&B OFFLINE]" if wandb else "")
    banner(f"STAGE {label}")
    csv = make_fishing_csv()
    out = SD / "ck_fishing" / "fishing_xgb.json"
    cmd = [
        "src/models/train_illegal_fishing.py", "--csv", csv,
        "--out", str(out), "--shap_out", str(SD / "ck_fishing" / "shap.png"),
    ]
    if not wandb:
        cmd.append("--no_wandb")
    else:
        cmd += ["--wandb_project", "maritime-smoke"]
    ok = run(cmd, env=extra_env)
    return ok and out.exists()


def stage_wandb():
    if not (os.environ.get("WANDB_API_KEY") or True):  # offline never needs a key
        pass
    banner("STAGE wandb — run M2 with W&B in OFFLINE mode (no login needed)")
    return stage_fishing(extra_env={"WANDB_MODE": "offline"}, wandb=True)


def stage_hf():
    banner("STAGE hf — real tiny push to smoke/ subfolder (only if HF_TOKEN is set)")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("HF_TOKEN not set -> skipping real push. "
              "Verifying the push helper FAILS GRACEFULLY instead...")
        code = ("from src.models.hf_utils import push_to_hub; "
                "ok = push_to_hub('does/not/exist.pt', 'smoke', repo_id='x/y'); "
                "assert ok is False; print('graceful-failure OK')")
        return run(["-c", code])
    # real push of a small file
    probe = SD / "hf_probe.txt"
    probe.write_text(f"smoke test {time.time()}")
    code = ("from src.models.hf_utils import push_to_hub, DEFAULT_HF_REPO; "
            f"ok = push_to_hub(r'{probe}', 'smoke', repo_id=DEFAULT_HF_REPO, private=True); "
            "assert ok; print('real push OK')")
    return run(["-c", code])


STAGE_FNS = {
    "imports": stage_imports, "preprocess": stage_preprocess, "oil": stage_oil,
    "detect": stage_detect, "fishing": stage_fishing, "wandb": stage_wandb, "hf": stage_hf,
}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Colab smoke test for the training stack")
    ap.add_argument("--stage", default="all", help=f"one of {ALL_STAGES} or 'all'")
    ap.add_argument("--skip", nargs="*", default=[], help="stages to skip")
    args = ap.parse_args()

    stages = ALL_STAGES if args.stage == "all" else [args.stage]
    stages = [s for s in stages if s not in args.skip]

    banner(f"SMOKE TEST — python {sys.version.split()[0]} | stages: {stages}")
    SD.mkdir(exist_ok=True)
    t0 = time.time()

    for s in stages:
        try:
            RESULTS[s] = STAGE_FNS[s]()
        except Exception as e:
            print(f"!! stage '{s}' raised {type(e).__name__}: {e}", flush=True)
            RESULTS[s] = False

    banner("SUMMARY")
    for s in stages:
        print(f"  {'PASS' if RESULTS.get(s) else 'FAIL'}  {s}")
    n_fail = sum(1 for s in stages if not RESULTS.get(s))
    print(f"\n{len(stages) - n_fail}/{len(stages)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
