"""Data: sort collected crops into labelled folders, or drop a photo to classify."""
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QMimeData, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget,
)

from jacket import config
from ren import dataset
from ren.theme import Fonts
from ren.widgets import Card, TextButton

THUMB_PX = 112
BIN_TIPS = {
    "warm": "jacket, hoodie, sweater, coat",
    "light": "t-shirt, shirt, short sleeves, tank top",
    "discard": "blurry, cut off, not a person",
}
SORT_TIP = "drag onto a bin; shift or cmd-click picks several. add crops with save crops on the live page"
PHOTO_TEXT = "drop a photo"
PHOTO_TIP = "everyone in it is labelled on the live page"


def image_paths(mime: QMimeData) -> list[Path]:
    """Local image files in a drag."""
    if not mime.hasUrls():
        return []
    paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
    return [path for path in paths if path.suffix.lower() in config.IMAGE_EXTENSIONS]


class CropGrid(QListWidget):
    """Thumbnails that drag out as files."""

    def __init__(self) -> None:
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(THUMB_PX, THUMB_PX))
        self.setGridSize(QSize(THUMB_PX + 16, THUMB_PX + 16))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_paths(self, paths: list[Path]) -> None:
        self.clear()
        for path in paths:
            pixmap = QPixmap(str(path)).scaled(THUMB_PX, THUMB_PX, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
            item = QListWidgetItem(QIcon(pixmap), "")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(path.name)
            self.addItem(item)

    def paths(self) -> list[Path]:
        return [Path(self.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.count())]

    def mimeData(self, items: list[QListWidgetItem]) -> QMimeData:
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(item.data(Qt.ItemDataRole.UserRole)) for item in items])
        return mime


class DropTarget(QFrame):
    """A dashed box that takes dropped image files."""

    dropped = Signal(list)  # list[Path]

    def __init__(self, text: str, fonts: Fonts, height: int) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(height)
        self._label = QLabel(text)
        self._label.setFont(fonts.text(config.UI_FONT_PX["small"], "medium"))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        self._set_hover(False)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def _set_hover(self, hover: bool) -> None:
        c = config.UI_COLORS
        fill, edge = (c["accent_soft"], c["accent"]) if hover else (c["background"], c["switch"])
        self.setStyleSheet(f"DropTarget {{ background: {fill}; border: 2px dashed {edge}; border-radius: 14px; }}"
                           f" QLabel {{ color: {c['label']}; }}")

    def dragEnterEvent(self, event) -> None:
        if image_paths(event.mimeData()):
            event.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, event) -> None:
        self._set_hover(False)

    def dropEvent(self, event) -> None:
        self._set_hover(False)
        paths = image_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(paths)


class DataPage(QWidget):
    photo_chosen = Signal(object)  # Path

    def __init__(self, fonts: Fonts, choose_photo: Callable[[QWidget], str] | None = None) -> None:
        super().__init__()
        self._fonts = fonts
        self._choose_photo = choose_photo or (
            lambda parent: QFileDialog.getOpenFileName(parent, "classify a photo", "", "Images (*.jpg *.jpeg *.png)")[0]
        )
        pad = config.UI_PAGE_PADDING_PX
        root = QHBoxLayout(self)
        root.setContentsMargins(pad, pad - 10, pad, pad - 10)
        root.setSpacing(pad - 10)

        sort_card = Card()
        sort = QVBoxLayout(sort_card.body())
        sort.setContentsMargins(28, 22, 28, 22)
        self._sort_title = self._title("to sort")
        self._sort_title.setToolTip(SORT_TIP)
        self.grid = CropGrid()
        self.grid.setToolTip(SORT_TIP)
        sort.addWidget(self._sort_title)
        sort.addSpacing(8)
        sort.addWidget(self.grid, 1)
        root.addWidget(sort_card, 3)

        side = QVBoxLayout()
        side.setSpacing(0)
        bins_card = Card()
        bins = QVBoxLayout(bins_card.body())
        bins.setContentsMargins(28, 22, 28, 22)
        bins.setSpacing(12)
        bins.addWidget(self._title("sort into"))
        self.bins: dict[str, DropTarget] = {}
        for name in dataset.BINS:
            target = DropTarget(self._bin_name(name), fonts, 56)
            target.setToolTip(BIN_TIPS[name])
            target.dropped.connect(lambda paths, n=name: self._file(paths, n))
            bins.addWidget(target)
            self.bins[name] = target
        side.addWidget(bins_card)

        photo_card = Card()
        photo = QVBoxLayout(photo_card.body())
        photo.setContentsMargins(28, 22, 28, 22)
        photo.setSpacing(12)
        photo.addWidget(self._title("classify"))
        self.photo_target = DropTarget(PHOTO_TEXT, fonts, 100)
        self.photo_target.setToolTip(PHOTO_TIP)
        self.photo_target.dropped.connect(lambda paths: self.photo_chosen.emit(paths[0]))
        browse = TextButton("choose", fonts)
        browse.setToolTip(PHOTO_TIP)
        browse.clicked.connect(self._browse)
        photo.addWidget(self.photo_target)
        photo.addWidget(browse, 0, Qt.AlignmentFlag.AlignLeft)
        side.addWidget(photo_card)
        side.addStretch(1)
        root.addLayout(side, 2)
        self.refresh()

    @staticmethod
    def _bin_name(name: str) -> str:
        """Clothing words on screen; the folders keep warm and light for jacket.eval."""
        return config.CLOTHING_NAMES.get(name, name)

    def _title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(self._fonts.text(config.UI_FONT_PX["title"], "bold"))
        return label

    def refresh(self) -> None:
        """Reread the folders; cheap enough to run on every visit."""
        self.grid.set_paths(dataset.images_in(config.RAW_CROPS_DIR))
        self._sort_title.setText(f"to sort  {self.grid.count()}")
        for name, target in self.bins.items():
            target.set_text(f"{self._bin_name(name)}  {len(dataset.images_in(dataset.bin_dir(name)))}")

    def showEvent(self, event) -> None:
        self.refresh()
        super().showEvent(event)

    def _file(self, paths: list[Path], name: str) -> None:
        dataset.file_into(paths, name)
        self.refresh()

    def _browse(self) -> None:
        chosen = self._choose_photo(self)
        if chosen:
            self.photo_chosen.emit(Path(chosen))
