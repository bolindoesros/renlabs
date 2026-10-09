import pytest

from jacket import camera
from jacket.camera import CameraError, find_camera_index


def test_finds_lowest_matching_index_ignoring_case(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: {0: "Acer FHD", 1: "Acer FHD", 4: "HD Pro Webcam C920", 5: "HD Pro Webcam C920"})
    assert find_camera_index("c920") == 4


def test_missing_camera_error_lists_what_exists(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: {0: "Acer FHD"})
    with pytest.raises(CameraError, match="Acer FHD"):
        find_camera_index("C920")
