import numpy as np

from jacket import config
from jacket.person_detector import torso_top_row

BOX_TOP, BOX_BOTTOM = 50, 470


EYE_X = {"left_eye": 480.0, "right_eye": 440.0}  # 40 px apart


def keypoints(**rows: float | None):
    """Build keypoints from named rows; None is missing."""
    names = {
        "nose": config.KEYPOINT_NOSE,
        "left_eye": config.KEYPOINT_LEFT_EYE,
        "right_eye": config.KEYPOINT_RIGHT_EYE,
        "left_shoulder": config.KEYPOINT_LEFT_SHOULDER,
        "right_shoulder": config.KEYPOINT_RIGHT_SHOULDER,
    }
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    for name, row in rows.items():
        if row is not None:
            xy[names[name], 1], conf[names[name]] = row, 0.9
            xy[names[name], 0] = EYE_X.get(name, 460.0)
    return xy, conf


def test_averages_two_shoulders():
    xy, conf = keypoints(left_shoulder=400, right_shoulder=420)
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) == 410


def test_uses_single_shoulder():
    xy, conf = keypoints(left_shoulder=400)
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) == 400


def test_chin_estimate_when_shoulders_missing():
    xy, conf = keypoints(nose=200, left_eye=180, right_eye=180)
    expected = int(200 + config.CHIN_BELOW_NOSE_EYE_DISTANCES * 40)
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) == expected


def test_takes_lower_of_shoulders_and_chin():
    xy, conf = keypoints(nose=200, left_eye=180, right_eye=180, left_shoulder=210)
    expected = int(200 + config.CHIN_BELOW_NOSE_EYE_DISTANCES * 40)
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) == expected


def test_none_without_confident_keypoints():
    xy, conf = keypoints()
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) is None


def test_none_when_outside_box():
    xy, conf = keypoints(left_shoulder=600)
    assert torso_top_row(xy, conf, BOX_TOP, BOX_BOTTOM) is None
