"""
Manually push any local checkpoint/folder to the project's Hugging Face repo.

Use this when you trained without --push_hf, or to re-upload after fixing auth.

Examples:
  # push the whole oil checkpoint dir into the oil/ subfolder
  python src/models/push_to_hf.py checkpoints/oil --subfolder oil

  # push a single YOLO weight
  python src/models/push_to_hf.py checkpoints/vessel/hrsid_yolov8m/weights/best.pt \
         --subfolder vessel/hrsid_yolov8m

  # push everything under checkpoints/ at once (mirrors local layout)
  python src/models/push_to_hf.py checkpoints --subfolder .

Auth: export HF_TOKEN=hf_xxx   (write token)   OR   huggingface-cli login
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.hf_utils import setup_logger, push_to_hub, DEFAULT_HF_REPO


def main():
    p = argparse.ArgumentParser(description="Push a local path to the HF Hub.")
    p.add_argument("path", help="Local file or folder to upload")
    p.add_argument("--subfolder", default=".",
                   help="Destination subfolder in the repo (e.g. oil / vessel / fishing)")
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO)
    p.add_argument("--hf_token", default=None,
                   help="HF write token (else HF_TOKEN env / cached login)")
    p.add_argument("--hf_public", action="store_true",
                   help="Make the repo public (default: private)")
    p.add_argument("--message", default=None, help="Commit message")
    args = p.parse_args()

    logger = setup_logger("hf-push")
    ok = push_to_hub(
        local_path=args.path,
        path_in_repo=args.subfolder,
        repo_id=args.hf_repo,
        token=args.hf_token,
        private=not args.hf_public,
        commit_message=args.message,
        logger=logger,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
