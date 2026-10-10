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


def test_other_platforms_fall_back_to_the_default_index(monkeypatch):
    from jacket import config
    monkeypatch.setattr(camera, "list_cameras", lambda: {})
    monkeypatch.setattr(camera.sys, "platform", "win32")
    assert find_camera_index("C920") == config.CAMERA_DEFAULT_INDEX


def test_linux_still_fails_loudly_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(camera, "list_cameras", lambda: {})
    monkeypatch.setattr(camera.sys, "platform", "linux")
    with pytest.raises(CameraError):
        find_camera_index("C920")


@pytest.mark.parametrize("platform,name", [("linux", "CAP_V4L2"), ("win32", "CAP_DSHOW"), ("darwin", "CAP_AVFOUNDATION"), ("sunos5", "CAP_ANY")])
def test_each_platform_gets_its_own_backend(monkeypatch, platform, name):
    import cv2
    monkeypatch.setattr(camera.sys, "platform", platform)
    assert camera.capture_backend() == getattr(cv2, name)


def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    import torch
    from jacket.person_detector import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda") == "cuda"


def test_stylesheet_icon_paths_use_forward_slashes():
    from ren.theme import build_stylesheet
    import re
    assert all("\\\\" not in path for path in re.findall(r'url\\("([^"]+)"\\)', build_stylesheet()))
