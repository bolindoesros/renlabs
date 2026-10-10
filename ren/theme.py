"""Look and feel: soft, pastel, Google-style."""
import logging
from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPainterPath

from jacket import config
from ren.plan import LABEL_TO_STATE

logger = logging.getLogger("ren.theme")
GREYSCALE_AA = QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias  # no colour fringes
ICONS = config.PROJECT_ROOT / "ren" / "assets" / "icons"
CHEVRON_FILE = ICONS / "chevron-down.svg"
CHECK_FILE = ICONS / "check.svg"
LINE_DARKEN = 135  # box and ring lines are darker than fills
SHADOW_LAYERS = ((1, 3, 14), (3, 4, 9), (6, 6, 5), (10, 8, 2))  # spread, drop, alpha


@dataclass(frozen=True)
class Fonts:
    text_family: str
    mono_families: list[str]

    def text(self, px: int, weight: str = "regular", letter_spacing: float = 0.0) -> QFont:
        font = QFont(self.text_family)
        font.setPixelSize(px)
        font.setStyleStrategy(GREYSCALE_AA)
        font.setVariableAxis(QFont.Tag("wght"), float(config.UI_WEIGHT[weight]))  # real weights, not faux bold
        if letter_spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
        return font

    def mono(self, px: int) -> QFont:
        font = QFont()
        font.setFamilies(self.mono_families)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(px)
        font.setStyleStrategy(GREYSCALE_AA)
        return font


def load_fonts() -> Fonts:
    """Register the bundled Google Sans. Needs a QApplication."""
    font_id = QFontDatabase.addApplicationFont(str(config.UI_FONT_FILE))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if config.UI_TEXT_FONT in families:
        text_family = config.UI_TEXT_FONT
    else:
        logger.warning("could not load %s; using the system sans-serif", config.UI_FONT_FILE)
        text_family = "sans-serif"
    return Fonts(text_family, list(config.UI_MONO_FONTS))


def color(name: str) -> QColor:
    return QColor(config.UI_COLORS[name])


def with_alpha(value: QColor, alpha: int) -> QColor:
    tinted = QColor(value)
    tinted.setAlpha(alpha)
    return tinted


def state_color(state: str) -> QColor:
    """Pastel fill for a seat state."""
    return QColor(config.SEAT_COLORS[state])


def state_line(state: str) -> QColor:
    """A darker state tone, for lines on photos."""
    return state_color(state).darker(LINE_DARKEN)


def label_color(label: str) -> QColor:
    """Fill colour for a clothing label."""
    return state_color(LABEL_TO_STATE[label])


def label_line(label: str) -> QColor:
    return state_line(LABEL_TO_STATE[label])


def paint_soft_shadow(painter: QPainter, rect: QRectF, radius: float) -> None:
    """Faint, growing rounded rects: a soft shadow."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    for spread, drop, alpha in SHADOW_LAYERS:
        painter.setBrush(with_alpha(color("shadow"), alpha))
        grown = rect.adjusted(-spread, -spread + drop, spread, spread + drop)
        painter.drawRoundedRect(grown, radius + spread, radius + spread)
    painter.restore()


def rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def build_stylesheet() -> str:
    c = config.UI_COLORS
    radius = config.UI_RADIUS["card"]
    return f"""
    QWidget {{ background: {c['page']}; color: {c['text']}; }}
    QLabel {{ background: transparent; }}
    QFrame#card, QFrame#card QWidget {{ background: {c['background']}; }}
    QFrame#card QLabel {{ background: transparent; }}
    QFrame#card {{ border-radius: {radius}px; border: 1px solid {c['hairline']}; }}
    QFrame#hairline {{ background: {c['hairline']}; border: none; min-height: 1px; max-height: 1px; }}
    QComboBox, QPushButton#viewButton {{
        background: {c['chip']}; color: {c['text']}; border: none; border-radius: 19px;
        padding: 7px 40px 7px 18px; min-height: 24px; text-align: left;
    }}
    QComboBox:hover, QPushButton#viewButton:hover {{ background: {c['hover']}; }}
    QComboBox:disabled, QPushButton#viewButton:disabled {{ color: {c['muted']}; background: {c['page']}; }}
    QComboBox::drop-down {{ border: none; width: 34px; }}
    QComboBox::down-arrow {{ image: url("{CHEVRON_FILE}"); width: 12px; height: 12px; }}
    QPushButton#viewButton::menu-indicator {{
        image: url("{CHEVRON_FILE}"); subcontrol-origin: padding; subcontrol-position: center right; right: 14px;
    }}
    QComboBox QAbstractItemView {{
        background: {c['background']}; border: 1px solid {c['hairline']}; border-radius: 14px;
        outline: 0; padding: 6px; selection-background-color: {c['accent_soft']}; selection-color: {c['text']};
    }}
    QLineEdit {{
        background: {c['chip']}; border: 1px solid transparent; border-radius: 19px;
        padding: 7px 16px; selection-background-color: {c['accent_soft']}; selection-color: {c['text']};
    }}
    QLineEdit:focus {{ border-color: {c['accent']}; background: {c['background']}; }}
    QMenu {{ background: {c['background']}; border: 1px solid {c['hairline']}; border-radius: 16px; padding: 8px; }}
    QMenu::item {{ padding: 8px 40px 8px 38px; border-radius: 10px; }}
    QMenu::item:selected {{ background: {c['hover']}; color: {c['text']}; }}
    QMenu::item:disabled {{ color: {c['muted']}; padding-top: 10px; }}
    QMenu::indicator {{ width: 14px; height: 14px; left: 14px; }}
    QMenu::indicator:checked {{ image: url("{CHECK_FILE}"); }}
    QMenu::separator {{ height: 1px; background: {c['hairline']}; margin: 6px 10px; }}
    QToolTip {{ background: #303134; color: #ffffff; border: none; border-radius: 6px; padding: 6px 10px; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {c['hairline']}; border-radius: 4px; min-height: 40px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    """
