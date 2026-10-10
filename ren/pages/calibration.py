"""Page for lining the seat grid up."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from jacket import config
from ren.calibration_view import CalibrationView
from ren.engine import FrameResult
from ren.plan import Calibration, PlanLayout, anchor_of
from ren.theme import Fonts
from ren.widgets import Card, TextButton


class CalibrationPage(QWidget):
    changed = Signal(object)  # Calibration
    finished = Signal()

    def __init__(self, fonts: Fonts, calibration: Calibration, layout: PlanLayout) -> None:
        super().__init__()
        pad = config.UI_PAGE_PADDING_PX
        root = QVBoxLayout(self)
        root.setContentsMargins(pad, pad - 6, pad, pad - 12)
        root.setSpacing(14)

        header = QHBoxLayout()
        self._title = title = QLabel("calibrate")
        title.setFont(fonts.text(config.UI_FONT_PX["title"], "bold"))
        hint = QLabel("drag the four corners onto the seating area")
        hint.setFont(fonts.text(config.UI_FONT_PX["body"]))
        hint.setStyleSheet(f"color: {config.UI_COLORS['label']};")
        header.addWidget(title)
        header.addSpacing(18)
        header.addWidget(hint)
        header.addStretch(1)
        self._reset = TextButton("reset", fonts)
        self._done = TextButton("done", fonts)
        header.addWidget(self._reset)
        header.addSpacing(8)
        header.addWidget(self._done)
        root.addLayout(header)

        self._view = CalibrationView(fonts)
        self._view.set_calibration(calibration)
        self._view.set_layout(layout)
        card = Card()
        inner = QVBoxLayout(card.body())
        inner.setContentsMargins(12, 12, 12, 12)
        inner.addWidget(self._view)
        root.addWidget(card, 1)

        self._view.changed.connect(self.changed)
        self._done.clicked.connect(self.finished)
        self._reset.clicked.connect(self._reset_corners)

    def _reset_corners(self) -> None:
        self._view.set_calibration(Calibration())
        self.changed.emit(Calibration())

    def set_venue(self, name: str) -> None:
        self._title.setText(f"calibrate {name}")

    def set_calibration(self, calibration: Calibration) -> None:
        self._view.set_calibration(calibration)

    def set_layout(self, layout: PlanLayout) -> None:
        self._view.set_layout(layout)

    def show_result(self, result: FrameResult) -> None:
        self._view.show_frame(result.frame, [anchor_of(box) for box in result.boxes])
