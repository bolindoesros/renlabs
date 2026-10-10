# PyInstaller spec; build with freeze/build.sh (macOS) or freeze\build.bat (Windows).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
APP_NAME = "Renlabs"

datas = [
    (str(ROOT / "ren" / "assets"), "ren/assets"),
    (str(ROOT / "weights"), "weights"),
    (str(ROOT / "freeze" / "cache" / "hf"), "hf"),  # from freeze/fetch_models.py
]
binaries = []
hiddenimports = collect_submodules("transformers.models.segformer")  # transformers loads models by name
for package in ("ultralytics", "open_clip", "timm", "torchvision"):  # data, lazy imports, and torchvision's _C_stable.so
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

analysis = Analysis(
    [str(ROOT / "freeze" / "entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tkinter"],
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,  # one folder, not one file: a 2 GB single file unpacks on every launch
    name=APP_NAME,
    console=False,
)
collect = COLLECT(exe, analysis.binaries, analysis.datas, name=APP_NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        collect,
        name=f"{APP_NAME}.app",
        bundle_identifier="com.renlabs.app",
        info_plist={
            "NSCameraUsageDescription": "Renlabs counts people in the room from the camera.",
            "NSHighResolutionCapable": True,
        },
    )
