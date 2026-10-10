"""Torso crops: what the clothing model sees, one tile per person."""
import math
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from jacket import config
from ren.engine import FrameResult
from ren.theme import Fonts, color, label_color, rounded
from ren.video_view import fit_rect, tag_text, to_qimage

TILE_GAP_PX = 12
CAPTION_PX = 30  # tag row under each crop
EMPTY_TEXT = "no people in view"


@dataclass(frozen=True)
class Tile:
    image: QImage | None  # None when the torso could not be cut
    text: str  # tag words, or why there is no crop
    fill: QColor | None  # tag colour; None for a plain caption


def tile_side(count: int, width: float, height: float) -> tuple[int, float]:
    """Columns and square side that make `count` tiles as large as possible."""
    best = (1, 0.0)
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        side = min(
            (width - TILE_GAP_PX * (cols - 1)) / cols,
            (height - TILE_GAP_PX * (rows - 1)) / rows - CAPTION_PX,
        )
        if side > best[1]:
            best = (cols, side)
    return best


class CropsView(QWidget):
    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._fonts = fonts
        self._tiles: list[Tile] = []
        self._message = EMPTY_TEXT
        self.setMinimumSize(160, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self) -> QSize:
        return QSize(240, 180)

    def tiles(self) -> list[Tile]:
        return self._tiles

    def show_result(self, result: FrameResult) -> None:
        tiles = []
        for index, (box, crop) in enumerate(zip(result.boxes, result.crops)):
            clothing = result.results[index] if result.results else None
            if crop.crop is None:
                tiles.append(Tile(None, crop.reason, None))
            else:
                fill = label_color(clothing.label) if clothing is not None else color("chip")
                tiles.append(Tile(to_qimage(crop.crop), tag_text(box, clothing), fill))
        self._tiles, self._message = tiles, "" if tiles else EMPTY_TEXT
        self.update()

    def show_message(self, text: str) -> None:
        self._tiles, self._message = [], text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        painter.fillRect(self.rect(), color("background"))
        if not self._tiles:
            painter.setFont(self._fonts.text(config.UI_FONT_PX["body"]))
            painter.setPen(color("muted"))
            painter.drawText(
                self.rect().adjusted(10, 8, -10, -8),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self._message,
            )
            return
        cols, side = tile_side(len(self._tiles), self.width(), self.height())
        side = max(side, 24.0)  # very cramped panels clip rather than vanish
        for index, tile in enumerate(self._tiles):
            row, col = divmod(index, cols)
            x = col * (side + TILE_GAP_PX)
            y = row * (side + CAPTION_PX + TILE_GAP_PX)
            self._paint_tile(painter, tile, QRectF(x, y, side, side))

    def _paint_tile(self, painter: QPainter, tile: Tile, frame: QRectF) -> None:
        radius = config.UI_RADIUS["viewport"] - 4
        painter.fillPath(rounded(frame, radius), color("viewport"))
        if tile.image is not None:
            target = fit_rect(tile.image.size(), frame)
            painter.save()
            painter.setClipPath(rounded(frame, radius))
            painter.drawImage(target, tile.image)
            painter.restore()
        font = self._fonts.text(config.UI_FONT_PX["tag"], "medium")
        painter.setFont(font)
        metrics = QFontMetrics(font)
        pad_x, pad_y = config.UI_TAG_PADDING_PX
        text = metrics.elidedText(tile.text, Qt.TextElideMode.ElideRight, int(frame.width() - 2 * pad_x))
        caption_top = frame.bottom() + 6
        if tile.fill is None:
            painter.setPen(color("muted"))
            painter.drawText(QRectF(frame.left(), caption_top, frame.width(), CAPTION_PX - 6),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, text)
            return
        tag = QRectF(frame.left(), caption_top, metrics.horizontalAdvance(text) + 2 * pad_x, metrics.height() + 2 * pad_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tile.fill)
        painter.drawRoundedRect(tag, config.UI_RADIUS["tag"], config.UI_RADIUS["tag"])
        painter.setPen(color("text"))
        painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, text)
