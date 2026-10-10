"""Shared test setup: free Qt widgets after each test, at a known moment.

Widgets left to Python's garbage collector get freed at random later points,
sometimes mid-way through building another widget, which crashes PySide6.
So automatic GC is off during tests and runs once per test file instead.
"""
import gc
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def collect_garbage_only_between_files():
    gc.disable()
    yield
    gc.enable()


@pytest.fixture(autouse=True, scope="module")
def collect_after_each_file():
    yield
    gc.collect()


@pytest.fixture(autouse=True)
def free_qt_widgets():
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # freed now, not by a later GC pass
