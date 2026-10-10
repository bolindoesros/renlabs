import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # A build ships its Hugging Face models (packaging/fetch_models.py) and never goes online.
    os.environ["HF_HUB_CACHE"] = str(Path(sys._MEIPASS) / "hf")
    os.environ["HF_HUB_OFFLINE"] = "1"

# Once the fashion model is downloaded, skip Hugging Face's online check at startup (~3s).
# Runs before any hf import; set HF_HUB_OFFLINE=0 to force a fresh download.
_hf_cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
if (_hf_cache.expanduser() / "models--Marqo--marqo-fashionSigLIP" / "snapshots").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
