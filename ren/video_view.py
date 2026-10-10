"""The video area: a rounded viewport with soft overlays."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from jacket import config
from jacket.types import Box, ClothingResult
from ren.engine import FrameResult, ViewSettings
from ren.plan import LABEL_TO_STATE, Line
from ren.theme import Fonts, color, label_color, label_line, rounded


def tag_text(box: Box, result: ClothingResult | None) -> str:
    """Tag words: hot, cold or unsure, plus confidence."""
    if result is None:
        return f"person {box.confidence:.0%}"
    state = LABEL_TO_STATE[result.label]
    if state == "unsure" or result.warm_prob is None:
        return state
    return f"{state} {max(result.warm_prob, 1 - result.warm_prob):.0%}"


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame = np.ascontiguousarray(frame_bgr)
    height, width = frame.shape[:2]
    return QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888).copy()


def fit_rect(image_size: QSize, available: QRectF) -> QRectF:
    """Largest aspect-correct rectangle, centred."""
    scale = min(available.width() / image_size.width(), available.height() / image_size.height())
    width, height = image_size.width() * scale, image_size.height() * scale
    return QRectF(
        available.left() + (available.width() - width) / 2,
        available.top() + (available.height() - height) / 2, width, height,
    )


class VideoView(QWidget):
    def __init__(self, fonts: Fonts, preferred_size: QSize = QSize(320, 180)) -> None:
        super().__init__()
        self._fonts = fonts
        self._preferred_size = preferred_size
        self._image: QImage | None = None
        self._strip: QImage | None = None  # torso crops, painted under the video
        self._boxes: list[Box] = []
        self._results: list[ClothingResult] = []
        self._message = "starting camera..."
        self._grid: list[Line] = []
        self._layers = ViewSettings()
        self._unseated: frozenset[int] = frozenset()
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self) -> QSize:
        return self._preferred_size

    def message(self) -> str:
        return self._message

    def show_result(self, result: FrameResult) -> None:
        self._image = to_qimage(result.frame)
        self._strip = to_qimage(result.crop_strip) if result.crop_strip is not None else None
        self._boxes, self._results, self._message = result.boxes, result.results, ""
        self.update()

    def show_message(self, text: str) -> None:
        self._image, self._strip, self._boxes, self._results, self._message = None, None, [], [], text
        self.update()

    def set_grid(self, lines: list[Line]) -> None:
        """Seat boundaries in image pixels; empty hides the grid."""
        self._grid = lines
        self.update()

    def set_layers(self, view: ViewSettings) -> None:
        """Which layers to draw: image, people, labels, shoulders, grid."""
        self._layers = view
        self.update()

    def set_unseated(self, indices) -> None:
        """People the seating plan could not place; drawn dashed."""
        self._unseated = frozenset(indices)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        painter.fillRect(self.rect(), color("background"))
        painter.fillPath(rounded(QRectF(self.rect()), config.UI_RADIUS["viewport"]), color("viewport"))
        if self._image is None:
            painter.setFont(self._fonts.text(config.UI_FONT_PX["body"]))
            painter.setPen(color("label"))
            painter.drawText(
                self.rect().adjusted(22, 20, -22, -20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, self._message,
            )
            return

        area = QRectF(self.rect())
        if self._strip is not None:
            area.setHeight(max(area.height() - self._strip.height() - config.UI_CROP_GAP_PX, 1))
        target = fit_rect(self._image.size(), area)
        painter.setClipPath(rounded(target, config.UI_RADIUS["viewport"] - 4))  # soft photo corners
        if self._layers.show_image:
            painter.drawImage(target, self._image)
        else:
            painter.fillRect(target, color("hairline"))
        scale = target.width() / self._image.width()
        if self._layers.show_grid:
            self._paint_grid(painter, target, scale)
        if self._layers.show_people:
            for index, box in enumerate(self._boxes):
                result = self._results[index] if self._results else None
                self._paint_person(painter, box, result, target, scale, index in self._unseated)
        painter.setClipping(False)
        if self._strip is not None:
            self._paint_strip(painter, target)

    def _paint_strip(self, painter: QPainter, target: QRectF) -> None:
        """Crops sit under the video; long strips are cut."""
        x, y = target.left(), target.bottom() + config.UI_CROP_GAP_PX
        visible = min(self._strip.width(), self.width() - x)
        painter.drawImage(QPointF(x, y), self._strip, QRectF(0, 0, visible, self._strip.height()))

    def _paint_grid(self, painter: QPainter, target: QRectF, scale: float) -> None:
        for (x1, y1), (x2, y2) in self._grid:
            start = QPointF(target.left() + x1 * scale, target.top() + y1 * scale)
            end = QPointF(target.left() + x2 * scale, target.top() + y2 * scale)
            painter.setPen(QPen(QColor(0, 0, 0, 70), 3))
            painter.drawLine(start, end)
            painter.setPen(QPen(QColor(255, 255, 255, 190), 1))
            painter.drawLine(start, end)

    def _paint_person(
        self, painter: QPainter, box: Box, result: ClothingResult | None, target: QRectF, scale: float,
        outside_plan: bool = False,
    ) -> None:
        if outside_plan:
            line, fill = color("muted"), color("hairline")
        elif result is not None:
            line, fill = label_line(result.label), label_color(result.label)
        else:
            line, fill = color("label"), color("chip")
        rect = QRectF(
            target.left() + box.x1 * scale, target.top() + box.y1 * scale,
            box.width * scale, box.height * scale,
        )
        radius = config.UI_RADIUS["box"]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 160), config.UI_BOX_HALO_PX))
        painter.drawRoundedRect(rect, radius, radius)
        style = Qt.PenStyle.DashLine if outside_plan else Qt.PenStyle.SolidLine
        painter.setPen(QPen(line, config.UI_BOX_LINE_PX, style))
        painter.drawRoundedRect(rect, radius, radius)
        if self._layers.show_torso_line and box.torso_top_y is not None:
            y = target.top() + box.torso_top_y * scale
            painter.setPen(QPen(line, 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        if self._layers.show_labels:
            text = "outside plan" if outside_plan else tag_text(box, result)
            self._paint_tag(painter, text, rect, target, fill)

    def _paint_tag(self, painter: QPainter, text: str, rect: QRectF, target: QRectF, ink: QColor) -> None:
        """Pastel tab on the box's top edge."""
        font = self._fonts.text(config.UI_FONT_PX["tag"], "medium")
        metrics = QFontMetrics(font)
        pad_x, pad_y = config.UI_TAG_PADDING_PX
        width, height = metrics.horizontalAdvance(text) + 2 * pad_x, metrics.height() + 2 * pad_y
        x = min(max(rect.left(), target.left()), target.right() - width)
        y = rect.top() - height
        if y < target.top():
            y = rect.top()
        tag = QRectF(x, y, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ink)
        painter.drawRoundedRect(tag, config.UI_RADIUS["tag"], config.UI_RADIUS["tag"])
        painter.setFont(font)
        painter.setPen(color("text"))
        painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, text)
