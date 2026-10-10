"""Seating plan panel: decision, plan and legend."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from jacket import config
from ren.decision import Decision, NeedWeights, VentSettings
from ren.engine import ViewSettings
from ren.plan import NEED_NAMES, PlanLayout, SeatingScene, Zone
from ren.seating_view import SeatingPlanView
from ren.theme import Fonts
from ren.widgets import TextButton, Legend

RULE_TIP = "people in shirts need air; people in jackets are fine and get less"
PHASE_VERBS = {"turning": "turning to", "opening": "opening towards", "aiming": "aiming at"}


def status_text(decision: Decision, scene: SeatingScene, people: int) -> str:
    """What the vent is doing, and why."""
    if people == 0:
        return "no one in view"
    if not scene.seats:
        return "everyone is outside the plan, check calibration"
    outside = f", {scene.off_plan} outside the plan" if scene.off_plan else ""
    if decision.closed:
        everyone_cold = all(reading.state == "cold" for reading in scene.seats.values())
        reason = "everyone seated is in a jacket" if everyone_cold else "nobody needs air"
        return f"vent shut, {reason}{outside}"
    text = f"{PHASE_VERBS.get(decision.phase, 'aiming at')} {decision.target_zone}"
    if not decision.reachable:
        text += ", out of reach"
    return text + outside


def legend_entries(weights: NeedWeights) -> list[tuple[str, str]]:
    return [(state, f"{NEED_NAMES[state]} {getattr(weights, state):g}") for state in ("hot", "unsure", "cold")]


def shrinkable(label: QLabel) -> QLabel:
    """Wrap long text so the panel can shrink."""
    label.setWordWrap(True)
    policy = label.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    label.setSizePolicy(policy)
    return label


class PlanPanel(QWidget):
    retry_requested = Signal()

    def __init__(self, fonts: Fonts) -> None:
        super().__init__()
        self._view = SeatingPlanView(fonts)
        self._error = QLabel()
        self._error.setFont(fonts.text(config.UI_FONT_PX["body"]))
        self._error.setStyleSheet(f"color: {config.UI_COLORS['error']};")
        self._error.setWordWrap(True)
        self._error.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        retry = TextButton("retry", fonts)
        retry.clicked.connect(self.retry_requested)
        failed = QWidget()
        failed_layout = QVBoxLayout(failed)
        failed_layout.setContentsMargins(16, 16, 16, 16)
        failed_layout.addWidget(self._error)
        failed_layout.addWidget(retry, 0, Qt.AlignmentFlag.AlignLeft)
        failed_layout.addStretch(1)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._view)
        self._stack.addWidget(failed)

        self._legend = Legend(legend_entries(NeedWeights()), fonts)
        self._legend.setToolTip(RULE_TIP)
        legend_caption = QLabel("air per seat")
        legend_caption.setFont(fonts.text(config.UI_FONT_PX["small"], "medium"))
        legend_caption.setStyleSheet(f"color: {config.UI_COLORS['muted']};")
        self._status = shrinkable(QLabel())
        self._status.setFont(fonts.text(config.UI_FONT_PX["headline"], "medium"))

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        top = QVBoxLayout()
        top.setContentsMargins(8, 2, 8, 4)
        top.addWidget(self._status)
        root.addLayout(top)
        root.addWidget(self._stack, 1)
        foot = QVBoxLayout()
        foot.setContentsMargins(8, 0, 8, 2)
        foot.setSpacing(2)
        foot.addWidget(legend_caption)
        foot.addWidget(self._legend)
        root.addLayout(foot)

    def set_state(
        self, layout: PlanLayout, scene: SeatingScene | None, decision: Decision | None,
        zones: list[Zone], vent: VentSettings, layers: ViewSettings,
    ) -> None:
        self._view.set_state(layout, scene, decision, zones, vent, layers)

    def set_weights(self, weights: NeedWeights) -> None:
        self._legend.set_entries(legend_entries(weights))

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def show_error(self, message: str) -> None:
        self._error.setText(f"the seating plan stopped\n\n{message}\n\nhide this panel to keep the camera running")
        self._stack.setCurrentIndex(1)

    def clear_error(self) -> None:
        self._stack.setCurrentIndex(0)

    def showing_error(self) -> bool:
        return self._stack.currentIndex() == 1

    def error_text(self) -> str:
        return self._error.text()
