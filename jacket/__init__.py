import os
from pathlib import Path

# Once the fashion model is downloaded, skip Hugging Face's online check at startup (~3s).
# Runs before any hf import; set HF_HUB_OFFLINE=0 to force a fresh download.
_hf_cache = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
if (_hf_cache.expanduser() / "models--Marqo--marqo-fashionSigLIP" / "snapshots").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
