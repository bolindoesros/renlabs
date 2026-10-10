import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QRectF, QSize
from PySide6.QtWidgets import QApplication

from jacket.types import Box, ClothingResult
from ren.engine import FrameResult, ViewSettings
from ren.theme import label_color, load_fonts
from ren.video_view import VideoView, fit_rect, to_qimage

GREY = 128


@pytest.fixture(scope="module")
def fonts():
    QApplication.instance() or QApplication([])
    return load_fonts()


def grey_frame(size: int = 100) -> np.ndarray:
    return np.full((size, size, 3), GREY, dtype=np.uint8)


def render(view: VideoView, size: int = 200):
    view.setMinimumSize(0, 0)  # a true square, so the frame fills it
    view.resize(size, size)
    return view.grab().toImage()


def redness(image, x: int, y: int) -> int:
    pixel = image.pixelColor(x, y)
    return pixel.red() - pixel.green()


def test_fit_rect_keeps_aspect_and_centres():
    rect = fit_rect(QSize(160, 90), QRectF(10, 20, 800, 800))
    assert rect.width() == 800 and rect.height() == pytest.approx(450)
    assert rect.left() == 10 and rect.top() == pytest.approx(20 + (800 - 450) / 2)


def test_fit_rect_is_height_limited_when_area_is_wide():
    rect = fit_rect(QSize(160, 90), QRectF(0, 0, 1000, 90))
    assert rect.height() == 90 and rect.width() == pytest.approx(160)
    assert rect.left() == pytest.approx((1000 - 160) / 2)


def test_bgr_frame_becomes_the_right_colours():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :] = (255, 0, 0)  # BGR blue
    pixel = to_qimage(frame).pixelColor(0, 0)
    assert (pixel.red(), pixel.green(), pixel.blue()) == (0, 0, 255)


def test_box_is_drawn_at_the_scaled_position(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=False, show_torso_line=False))
    box = Box(20, 20, 60, 60, 0.9, "person")  # frame 100px shown at 200px -> scale 2
    view.show_result(FrameResult(grey_frame(), [box], [ClothingResult("light", 0.1, "")], (), "1 of 1", ""))
    image = render(view)
    left_edge = max(redness(image, 39, 80), redness(image, 40, 80))
    assert left_edge > 40  # hot red line at x = 40
    assert abs(redness(image, 80, 80)) < 5  # inside the box the frame is untouched


def test_label_tag_is_drawn_above_the_box_in_the_label_colour(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=True, show_torso_line=False))
    box = Box(20, 40, 60, 80, 0.9, "person")
    view.show_result(FrameResult(grey_frame(), [box], [ClothingResult("warm", 0.9, "")], (), "1 of 1", ""))
    image = render(view)
    expected = label_color("warm")
    pixel = image.pixelColor(43, 70)  # tag padding; box top at y = 80
    assert (pixel.red(), pixel.green(), pixel.blue()) == (expected.red(), expected.green(), expected.blue())


def test_labels_off_leaves_no_tag(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=False, show_torso_line=False))
    box = Box(20, 40, 60, 80, 0.9, "person")
    view.show_result(FrameResult(grey_frame(), [box], [ClothingResult("warm", 0.9, "")], (), "1 of 1", ""))
    assert abs(redness(render(view), 43, 70)) < 5


def test_tag_moves_inside_when_box_touches_the_top(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=True, show_torso_line=False))
    box = Box(20, 0, 60, 60, 0.9, "person")  # no room above
    view.show_result(FrameResult(grey_frame(), [box], [ClothingResult("warm", 0.9, "")], (), "1 of 1", ""))
    expected = label_color("warm")
    pixel = render(view).pixelColor(44, 13)  # pill padding, left of the text
    assert (pixel.red(), pixel.green(), pixel.blue()) == (expected.red(), expected.green(), expected.blue())


def test_unclassified_person_gets_a_neutral_box(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=False, show_torso_line=False))
    box = Box(20, 20, 60, 60, 0.9, "person")
    view.show_result(FrameResult(grey_frame(), [box], [], (), "1", "person detected"))
    image = render(view)
    assert min(image.pixelColor(40, 80).red(), image.pixelColor(39, 80).red()) < GREY - 30  # dark line


def test_message_replaces_the_picture(fonts):
    view = VideoView(fonts)
    view.show_result(FrameResult(grey_frame(), [], [], (), "0", ""))
    view.show_message("no video")
    assert view.message() == "no video" and not render(view).isNull()


def test_tag_words_are_hot_cold_or_unsure():
    from ren.video_view import tag_text
    box = Box(0, 0, 10, 10, 0.86, "person")
    assert tag_text(box, ClothingResult("light", 0.08, "")) == "hot 92%"
    assert tag_text(box, ClothingResult("warm", 0.97, "")) == "cold 97%"
    assert tag_text(box, ClothingResult("unknown", 0.5, "between")) == "unsure"
    assert tag_text(box, None) == "person 86%"


def test_grid_lines_are_painted_over_the_picture(fonts):
    view = VideoView(fonts)
    view.set_layers(ViewSettings(show_labels=False, show_torso_line=False))
    view.show_result(FrameResult(grey_frame(), [], [], (), "0", ""))
    plain = render(view)
    view.set_grid([((0.0, 50.0), (100.0, 50.0))])  # a horizontal line across the frame
    gridded = render(view)
    assert plain.pixelColor(100, 100) != gridded.pixelColor(100, 100)


def one_person_view(fonts, layers: ViewSettings, results=None, unseated=()):
    view = VideoView(fonts)
    view.set_layers(layers)
    box = Box(20, 20, 60, 60, 0.9, "person")
    view.show_result(FrameResult(grey_frame(), [box], results if results is not None else [ClothingResult("light", 0.1, "")], (), "", ""))
    view.set_unseated(unseated)
    return view


def left_edge_redness(view):
    image = render(view)
    return max(redness(image, x, 80) for x in (39, 40))


def test_hiding_the_people_layer_removes_the_boxes(fonts):
    shown = one_person_view(fonts, ViewSettings(show_labels=False, show_people=True))
    hidden = one_person_view(fonts, ViewSettings(show_labels=False, show_people=False))
    assert left_edge_redness(shown) > 40 and abs(left_edge_redness(hidden)) < 5


def test_hiding_the_image_layer_keeps_the_boxes_but_not_the_picture(fonts):
    with_image = one_person_view(fonts, ViewSettings(show_labels=False, show_image=True))
    without = one_person_view(fonts, ViewSettings(show_labels=False, show_image=False))
    centre = (100, 150)
    assert render(with_image).pixelColor(*centre) != render(without).pixelColor(*centre)
    assert left_edge_redness(without) > 40


def edge_spread(view) -> int:
    """Green-channel spread along the left edge; dashes vary."""
    image = render(view)
    values = [image.pixelColor(40, y).green() for y in range(70, 110)]
    return max(values) - min(values)


def test_people_the_plan_could_not_place_are_drawn_dashed_and_muted(fonts):
    layers = ViewSettings(show_labels=False)
    solid = one_person_view(fonts, layers)
    dashed = one_person_view(fonts, layers, unseated=(0,))
    assert edge_spread(solid) < 15  # a continuous line
    assert edge_spread(dashed) > 30  # dashes with gaps
    image = render(dashed)
    assert all(image.pixelColor(40, y).red() - image.pixelColor(40, y).green() < 30 for y in range(70, 110))  # muted, not hot red


def test_unseated_people_are_tagged_outside_plan(fonts):
    view = one_person_view(fonts, ViewSettings(show_labels=True), unseated=(0,))
    assert not render(view).isNull()  # painting the tag must not fail
    assert view._unseated == frozenset({0})
