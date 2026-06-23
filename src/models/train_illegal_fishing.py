"""
Module 2 — Illegal Fishing Detection.
XGBoost on GFW pre-labeled fishing events (fishing.potentialRisk = target label).

Data source: GFW /v3/events?types[0]=FISHING, dataset=public-global-fishing-events:latest
Pre-downloaded to: /content/gfw_data/fishing_events_5k.csv (or fetch live via ais_matching.py)

Features used (all available directly from GFW response — no raw AIS engineering needed):
  fishing.averageSpeedKnots, fishing.totalDistanceKm, fishing.averageDurationHours,
  distances.startDistanceFromPortKm, distances.startDistanceFromShoreKm,
  regions.mpa (in MPA?), regions.highSeas (in high seas?),
  fishing.vesselPublicAuthorizationStatus (encoded), vessel.flag (encoded)

Target: fishing.potentialRisk (True/False, pre-labeled by GFW)
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_fscore_support)
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.explain.shap_explain import explain_xgboost, get_top_factors
from src.models.hf_utils import (
    setup_logger, log_banner, push_to_hub, add_hf_args, init_wandb, add_wandb_args,
)


# ── GFW data loader ───────────────────────────────────────────────────────────

def load_gfw_fishing_features(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load GFW fishing events CSV and build XGBoost-ready feature matrix.
    Returns (X, y) where y = fishing.potentialRisk (1=risk, 0=safe).

    Expected CSV: output of fetch_gfw_events("FISHING", ...) saved to disk.
    All features are direct GFW columns — no trajectory engineering needed.
    """
    df = pd.read_csv(csv_path)

    # ── numeric features (direct from GFW) ──
    num_cols = [
        "fishing.averageSpeedKnots",
        "fishing.totalDistanceKm",
        "fishing.averageDurationHours",
        "distances.startDistanceFromPortKm",
        "distances.startDistanceFromShoreKm",
        "distances.endDistanceFromPortKm",
        "distances.endDistanceFromShoreKm",
    ]
    X = df[[c for c in num_cols if c in df.columns]].copy()
    X = X.fillna(X.median())

    # ── binary zone flags ──
    X["in_mpa"]        = df["regions.mpa"].apply(
        lambda x: 0 if (pd.isna(x) or str(x) in ["[]", "nan"]) else 1
    )
    X["in_high_seas"]  = df["regions.highSeas"].apply(
        lambda x: 0 if (pd.isna(x) or str(x) in ["[]", "nan"]) else 1
    )
    X["in_rfmo"]       = df["regions.rfmo"].apply(
        lambda x: 0 if (pd.isna(x) or str(x) in ["[]", "nan"]) else 1
    ) if "regions.rfmo" in df.columns else 0

    # ── categorical: authorization status (3 levels) ──
    auth_map = {"publicly_authorized": 0, "partially_matched": 1, "unmatched": 2}
    X["auth_status"] = df.get("fishing.vesselPublicAuthorizationStatus", "unmatched").map(
        lambda x: auth_map.get(str(x), 2)
    )

    # ── categorical: flag state (top-20 flags encoded, rest = 99) ──
    if "vessel.flag" in df.columns:
        top_flags = df["vessel.flag"].value_counts().nlargest(20).index.tolist()
        flag_enc  = {f: i for i, f in enumerate(top_flags)}
        X["flag_enc"] = df["vessel.flag"].map(lambda x: flag_enc.get(str(x), 99))
    else:
        X["flag_enc"] = 99

    # ── target label ──
    y = df["fishing.potentialRisk"].map(
        lambda x: 1 if str(x).lower() in ["true", "1"] else 0
    ) if "fishing.potentialRisk" in df.columns else pd.Series([0] * len(df))

    print(f"[load_gfw_fishing_features] {len(X)} samples | "
          f"risk={y.sum()} ({100*y.mean():.1f}%) | features={X.columns.tolist()}")
    # ⚠️ LEAKAGE CAVEAT: GFW's `potentialRisk` label is itself partly derived from
    # authorization status and zone — which are also fed in as features (auth_status,
    # in_mpa, in_high_seas). High AUC here may reflect the model re-learning GFW's own
    # rule rather than independent signal. Report this honestly; if you want a cleaner
    # test of learned behaviour, drop `auth_status` and re-evaluate.
    if 100 * y.mean() < 2 or 100 * y.mean() > 98:
        print("  ⚠️ Highly imbalanced/near-degenerate label — metrics will be fragile; "
              "verify the event pull before trusting results.")
    return X, y


# ── Legacy feature engineering (raw AIS tracks, kept as fallback) ──────────────

def engineer_features(ais_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build interpretable trajectory features from raw AIS track.
    These feed XGBoost and are the basis for SHAP explanations.
    """
    feats = pd.DataFrame()
    g = ais_df.sort_values("timestamp").groupby("mmsi")

    feats["mean_speed"]          = g["speed"].mean()
    feats["std_speed"]           = g["speed"].std().fillna(0)
    feats["min_speed"]           = g["speed"].min()
    feats["max_speed"]           = g["speed"].max()

    # heading variance — fishing vessels zigzag a lot
    feats["heading_variance"]    = g["heading"].var().fillna(0)

    # time in zone (fraction of pings inside fishing grounds)
    if "in_fishing_zone" in ais_df.columns:
        feats["frac_in_zone"]    = g["in_fishing_zone"].mean()
    else:
        feats["frac_in_zone"]    = 0.0

    if "in_mpa" in ais_df.columns:
        feats["frac_in_mpa"]     = g["in_mpa"].mean()
    else:
        feats["frac_in_mpa"]     = 0.0

    # loitering = slow + heading change
    if "speed" in ais_df.columns:
        ais_df["loitering"] = (ais_df["speed"] < 2.0).astype(int)
        feats["loiter_frac"]     = g["loitering"].mean()
    else:
        feats["loiter_frac"]     = 0.0

    # AIS silence gaps
    ais_df["ts"] = pd.to_datetime(ais_df["timestamp"])
    ais_df["gap_sec"] = g["ts"].diff().dt.total_seconds()
    feats["max_ais_gap_h"]       = g["gap_sec"].max().fillna(0) / 3600
    feats["mean_ais_gap_h"]      = g["gap_sec"].mean().fillna(0) / 3600

    # track duration
    feats["track_duration_h"]    = (g["ts"].max() - g["ts"].min()).dt.total_seconds() / 3600

    if "vessel_type" in ais_df.columns:
        le = LabelEncoder()
        ais_df["vessel_type_enc"] = le.fit_transform(ais_df["vessel_type"].fillna("unknown"))
        feats["vessel_type_enc"] = g["vessel_type_enc"].first()

    return feats.reset_index()


# ── XGBoost model ─────────────────────────────────────────────────────────────

def train_fishing_classifier(
    feats: pd.DataFrame,
    labels: pd.Series,          # 1 = potentialRisk / 0 = no risk (GFW label)
    save_path: str = "checkpoints/fishing/fishing_xgb.json",
    shap_plot_path: str = "checkpoints/fishing/shap_fishing.png",
    logger=None,
    wandb_run=None,             # optional active W&B run for live + final logging
):
    log = logger or setup_logger("fishing-train")
    X = feats.drop(columns=["mmsi"], errors="ignore")
    y = labels

    # ── guard against degenerate labels (GFW potentialRisk can be all-one-class) ──
    if y.nunique() < 2:
        raise ValueError(
            f"Target has a single class (value={y.unique().tolist()}, "
            f"positives={int(y.sum())}/{len(y)}). The downloaded GFW events are not "
            "usable for binary classification — pull a more balanced set or change the target."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    log_banner(log, "M2 ILLEGAL FISHING — XGBoost", {
        "train rows":       len(X_train),
        "test rows":        len(X_test),
        "features":         X.shape[1],
        "train pos/neg":    f"{n_pos}/{n_neg}",
        "scale_pos_weight": round(scale_pos_weight, 3),
        "feature names":    X.columns.tolist(),
    })

    # Optional: log per-round aucpr to W&B via xgboost's native callback.
    xgb_callbacks = []
    if wandb_run is not None:
        try:
            from wandb.integration.xgboost import WandbCallback
            xgb_callbacks.append(WandbCallback(log_model=False))
            log.info("W&B XGBoost callback attached (per-round metrics will stream).")
        except Exception as e:
            log.warning(f"W&B xgboost callback unavailable: {e}")

    # XGBoost >= 2.0: early_stopping_rounds is a CONSTRUCTOR arg (removed from fit());
    # use_label_encoder was removed entirely.
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        callbacks=xgb_callbacks or None,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    log.info(f"Fit complete in {time.time() - t0:.1f}s "
             f"(best_iteration={getattr(model, 'best_iteration', 'n/a')})")

    preds   = model.predict(X_test)
    probs   = model.predict_proba(X_test)[:, 1]
    p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    auc = roc_auc_score(y_test, probs)

    log_banner(log, "RESULTS", {
        "Precision": f"{p:.4f}", "Recall": f"{r:.4f}",
        "F1": f"{f:.4f}", "ROC-AUC": f"{auc:.4f}",
    })
    log.info("\n" + classification_report(y_test, preds))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_path))
    log.info(f"✓ Model saved → {save_path}")

    # SHAP
    shap_vals = explain_xgboost(model, X_test, feature_names=X.columns.tolist(),
                                 save_path=str(shap_plot_path))
    top = get_top_factors(shap_vals, X.columns.tolist(), top_n=5)
    log.info(f"✓ SHAP plot → {shap_plot_path}")
    log.info(f"Top SHAP factors (%): {json.dumps(top)}")

    metrics = {"precision": p, "recall": r, "f1": f, "roc_auc": auc, "shap_factors": top}
    metrics_path = save_path.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in metrics.items()}, fp, indent=2)
    log.info(f"✓ Metrics → {metrics_path}")

    # ── W&B: log final metrics + SHAP plot + feature-importance table ──
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.log({"precision": p, "recall": r, "f1": f, "roc_auc": auc})
            if Path(shap_plot_path).exists():
                wandb_run.log({"shap_summary": wandb.Image(str(shap_plot_path))})
            tbl = wandb.Table(columns=["feature", "shap_pct"],
                              data=[[k, v] for k, v in top.items()])
            wandb_run.log({"top_shap_factors": tbl})
        except Exception as e:
            log.warning(f"W&B logging skipped: {e}")
    return model, metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",      default="data/gfw/fishing_events_5k.csv",
                        help="Path to GFW fishing events CSV")
    parser.add_argument("--out",      default="checkpoints/fishing/fishing_xgb.json")
    parser.add_argument("--shap_out", default="checkpoints/fishing/shap_fishing.png")
    add_wandb_args(parser, default_project="maritime-fishing")
    add_hf_args(parser)
    args = parser.parse_args()

    logger = setup_logger("fishing-train",
                          logfile=str(Path(args.out).parent / "train_fishing.log"))
    logger.info(f"Loading GFW fishing events from {args.csv}")
    X, y = load_gfw_fishing_features(args.csv)

    run = init_wandb(args.wandb_project, "fishing_xgb",
                     config={"csv": args.csv, "n_samples": len(X), "n_features": X.shape[1],
                             "pos_rate": float(y.mean())},
                     entity=args.wandb_entity, enabled=not args.no_wandb, logger=logger)
    model, metrics = train_fishing_classifier(X, y, args.out, args.shap_out,
                                              logger=logger, wandb_run=run)
    if run is not None:
        run.finish()

    if args.push_hf:
        out_dir = Path(args.out).parent          # checkpoints/fishing/
        logger.info(f"Pushing fishing artifacts to HF '{args.hf_repo}' (subfolder: fishing)...")
        push_to_hub(str(out_dir), path_in_repo="fishing", repo_id=args.hf_repo,
                    token=args.hf_token, private=not args.hf_public,
                    commit_message=f"fishing: F1={metrics['f1']:.4f} AUC={metrics['roc_auc']:.4f}",
                    logger=logger)
