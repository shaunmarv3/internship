"""
Shared helpers for the training scripts:
  1. setup_logger()  — consistent, timestamped console + file logging
  2. log_banner()    — readable config/section banners
  3. gpu_info()      — device + GPU memory summary
  4. push_to_hub()   — upload trained artifacts into ONE private HF repo,
                       organized in subfolders (oil/ , vessel/ , fishing/).

HF auth (any one of these works):
  - export HF_TOKEN=hf_xxx           (or pass --hf_token)
  - huggingface-cli login           (cached token; pass token=None)
  - from huggingface_hub import login; login()   # in a Colab cell

Nothing here is allowed to crash a finished training run — push failures are
logged and swallowed, because the checkpoint is already safe on local disk.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Single private repo for the whole project; models live in subfolders.
DEFAULT_HF_REPO = "shaunmarvell/maritime-security-intelligence"


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str = "train", logfile: Optional[str] = None) -> logging.Logger:
    """Console (stdout) + optional file logger with timestamps. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:                      # already configured this run
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Make stdout tolerant of non-UTF8 consoles (Windows cp1252) so stray glyphs
    # in log messages never raise UnicodeEncodeError. Colab/Linux are UTF-8 anyway.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    console_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(console_fmt)
    logger.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(fh)
        logger.info(f"Logging to file: {logfile}")

    return logger


def log_banner(logger: logging.Logger, title: str, items: dict = None):
    """Print a titled, boxed section so runs are easy to scan in a long log."""
    line = "=" * 64
    logger.info(line)
    logger.info(f"  {title}")
    if items:
        logger.info("-" * 64)
        width = max((len(str(k)) for k in items), default=0)
        for k, v in items.items():
            logger.info(f"  {str(k):<{width}} : {v}")
    logger.info(line)


def gpu_info() -> dict:
    """Human-readable device summary for the run banner."""
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            return {"device": "cuda", "gpu": name, "vram_gb": f"{total:.1f}",
                    "torch": torch.__version__}
        return {"device": "cpu", "torch": torch.__version__}
    except Exception as e:           # torch not importable (e.g. xgboost-only run)
        return {"device": "unknown", "note": str(e)}


def fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# ── Hugging Face Hub push ────────────────────────────────────────────────────

def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """CLI flag → HF_TOKEN → HUGGINGFACE_TOKEN → cached login (None)."""
    return (token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or None)


def push_to_hub(
    local_path: str,
    path_in_repo: str,
    repo_id: str = DEFAULT_HF_REPO,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Upload a file or folder into `repo_id` under `path_in_repo` (e.g. "oil").
    Creates the repo (private by default) if it does not exist.
    Returns True on success, False on any failure (never raises).
    """
    log = logger or setup_logger("hf")
    src = Path(local_path)
    if not src.exists():
        log.error(f"[HF] Nothing to push — path does not exist: {src}")
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("[HF] huggingface_hub not installed. `pip install huggingface-hub`")
        return False

    tok = resolve_hf_token(token)
    api = HfApi(token=tok)
    commit_message = commit_message or f"Add {path_in_repo} artifacts"

    try:
        api.create_repo(repo_id=repo_id, repo_type="model",
                        private=private, exist_ok=True)
        log.info(f"[HF] Repo ready: https://huggingface.co/{repo_id} "
                 f"({'private' if private else 'public'})")

        if src.is_dir():
            api.upload_folder(
                folder_path=str(src),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.tmp", "__pycache__/*", "*.lock"],
            )
            log.info(f"[HF] ✓ Uploaded folder '{src}' → {repo_id}/{path_in_repo}/")
        else:
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"{path_in_repo}/{src.name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            log.info(f"[HF] ✓ Uploaded file  '{src.name}' → {repo_id}/{path_in_repo}/{src.name}")
        return True

    except Exception as e:
        log.error(f"[HF] Push FAILED ({type(e).__name__}): {e}")
        log.error("[HF] Model is safe locally. To fix auth: set HF_TOKEN env var, "
                  "or run `huggingface-cli login` (needs WRITE token), then re-push "
                  "with src/models/push_to_hf.py")
        return False


def init_wandb(project: str, name: str, config: dict = None,
               entity: Optional[str] = None, enabled: bool = True,
               logger: Optional[logging.Logger] = None):
    """
    Start a W&B run safely. Returns the run, or None if disabled / unavailable /
    not logged in — training continues regardless (never blocks or raises).

    Auth on Lightning.ai: `wandb login` once, or set WANDB_API_KEY in the env.
    Offline fallback: set WANDB_MODE=offline to log locally without a network.
    """
    log = logger or setup_logger("wandb")
    if not enabled:
        log.info("W&B disabled (--no_wandb).")
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed — skipping W&B (`pip install wandb`).")
        return None
    try:
        run = wandb.init(project=project, name=name, entity=entity,
                         config=config or {}, reinit=True)
        log.info(f"W&B run live: {run.url}")
        return run
    except Exception as e:
        log.warning(f"W&B init failed ({type(e).__name__}: {e}) — continuing without it. "
                    "Run `wandb login` or set WANDB_API_KEY (or WANDB_MODE=offline).")
        return None


def add_wandb_args(parser, default_project: str):
    """Attach standard W&B flags."""
    parser.add_argument("--wandb_project", default=default_project)
    parser.add_argument("--wandb_entity", default=None,
                        help="W&B team/entity (else your default; or WANDB_ENTITY env)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser


def add_hf_args(parser):
    """Attach the standard HF push flags to an argparse parser."""
    parser.add_argument("--push_hf", action="store_true",
                        help="Upload the trained checkpoint to the Hugging Face Hub")
    parser.add_argument("--hf_repo", default=DEFAULT_HF_REPO,
                        help=f"HF repo id (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--hf_token", default=None,
                        help="HF write token (else uses HF_TOKEN env or cached login)")
    parser.add_argument("--hf_public", action="store_true",
                        help="Make the HF repo public (default: private)")
    return parser
