"""Download every model a build ships. Run: python -m freeze.fetch_models"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_CACHE = ROOT / "freeze" / "cache" / "download"  # a normal Hugging Face cache, full of symlinks
BUNDLE_CACHE = ROOT / "freeze" / "cache" / "hf"  # the same cache with real files; the build ships this
os.environ["HF_HUB_CACHE"] = str(DOWNLOAD_CACHE)  # empty cache, so it ends up holding exactly what the app loads
os.environ["HF_HUB_OFFLINE"] = "0"

from jacket import config  # noqa: E402
from jacket.classifier import build_classifier  # noqa: E402
from jacket.head_detector import ensure_weights  # noqa: E402
from ultralytics.utils.downloads import attempt_download_asset  # noqa: E402


def flatten(source: Path, target: Path) -> None:
    """Copy each repo with symlinks resolved; blobs are not needed once snapshots hold the files."""
    shutil.rmtree(target, ignore_errors=True)
    for repo in sorted(source.glob("models--*")):
        for part in ("refs", "snapshots", ".no_exist"):
            if (repo / part).is_dir():
                shutil.copytree(repo / part, target / repo.name / part, symlinks=False)
        print(f"bundled {repo.name}")


def main() -> None:
    config.WEIGHTS_DIR.mkdir(exist_ok=True)
    attempt_download_asset(str(config.YOLO_WEIGHTS_PATH))
    ensure_weights(config.HEAD_WEIGHTS_PATH, config.HEAD_WEIGHTS_URL, config.HEAD_WEIGHTS_SHA256)
    for model_key in config.CLOTHING_MODELS:
        build_classifier(model_key, "cpu")
    flatten(DOWNLOAD_CACHE, BUNDLE_CACHE)


if __name__ == "__main__":
    main()
