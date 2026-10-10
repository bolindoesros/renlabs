"""Build entry point; a windowed build has no console, so output goes to a log file."""
import multiprocessing
import sys

from jacket import config

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.stdout is None or sys.stderr is None:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = open(config.DATA_DIR / "renlabs.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    from ren.ui import main

    sys.exit(main())
