import dataclasses
import json
import logging
from dataclasses import replace

import pytest

from ren.decision import NeedWeights
from ren.plan import Calibration, PlanLayout
from ren.settings import AppSettings, load_settings, sanitize, save_settings
from ren.venues import Venue


def test_defaults_round_trip_through_disk(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(AppSettings(), path)
    assert load_settings(path) == AppSettings()


def test_changed_values_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    changed = AppSettings(
        view=replace(AppSettings().view, model_key="clip", mirror=False),
        decision=replace(AppSettings().decision, aim_mode="focus", weights=NeedWeights(hot=0.9, unsure=0.4, cold=0.1)),
        venues=(Venue("lt1", PlanLayout(rows=5, cols=9), Calibration((0.2, 0.3), (0.8, 0.3), (0.9, 0.9), (0.1, 0.9))), Venue("lt2")),
        venue="lt2",
    )
    save_settings(changed, path)
    assert load_settings(path) == changed


def test_missing_file_means_defaults(tmp_path):
    assert load_settings(tmp_path / "nope.json") == AppSettings()


def test_missing_keys_keep_their_defaults(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"venues": [{"name": "lt1", "layout": {"rows": 7}}]}))
    loaded = load_settings(path)
    assert loaded.layout == PlanLayout(rows=7, cols=PlanLayout().cols)
    assert loaded.view == AppSettings().view


def test_unknown_keys_are_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"venues": [{"name": "lt1", "layout": {"rows": 4, "banana": 1}}], "future": True}))
    with caplog.at_level(logging.WARNING, logger="ren.settings"):
        loaded = load_settings(path)
    assert loaded.layout.rows == 4
    assert "banana" in caplog.text and "future" in caplog.text


def test_wrong_type_keeps_the_default_and_says_so(tmp_path, caplog):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"venues": [{"name": "lt1", "layout": {"rows": "four", "cols": 8}}]}))
    with caplog.at_level(logging.WARNING, logger="ren.settings"):
        loaded = load_settings(path)
    assert loaded.layout.rows == PlanLayout().rows and loaded.layout.cols == 8
    assert "rows" in caplog.text


def test_a_bool_is_not_accepted_as_a_number(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"venues": [{"name": "lt1", "layout": {"rows": True}}]}))
    assert load_settings(path).layout.rows == PlanLayout().rows


def test_broken_json_falls_back_loudly(tmp_path, caplog):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    with caplog.at_level(logging.ERROR, logger="ren.settings"):
        assert load_settings(path) == AppSettings()
    assert "unreadable" in caplog.text


def test_top_level_list_falls_back_loudly(tmp_path, caplog):
    path = tmp_path / "s.json"
    path.write_text("[1, 2]")
    with caplog.at_level(logging.ERROR, logger="ren.settings"):
        assert load_settings(path) == AppSettings()


def test_save_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "deep" / "s.json"
    save_settings(AppSettings(), path)
    assert [p.name for p in path.parent.iterdir()] == ["s.json"]


def test_sanitize_clamps_out_of_range_values(caplog):
    wild = AppSettings(
        decision=replace(AppSettings().decision, focus_margin=-1.0),
        venues=(Venue("lt1", PlanLayout(rows=500, cols=0), Calibration(), replace(AppSettings().vent, tilt_max_deg=500.0, u=3.0)),),
    )
    with caplog.at_level(logging.WARNING, logger="ren.settings"):
        clean = sanitize(wild)
    assert clean.layout == PlanLayout(rows=12, cols=1)
    assert clean.vent.tilt_max_deg == 80.0 and clean.vent.u == 1.0 and clean.decision.focus_margin == 0.0
    assert "out of range" in caplog.text


def test_sanitize_repairs_a_bad_calibration_and_model():
    broken = AppSettings(
        view=replace(AppSettings().view, model_key="gpt-9"),
        venues=(Venue("lt1", PlanLayout(), Calibration((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5))),),
    )
    clean = sanitize(broken)
    assert clean.calibration == Calibration() and clean.view.model_key == AppSettings().view.model_key


def test_sanitize_repairs_an_unknown_detector():
    clean = sanitize(AppSettings(view=replace(AppSettings().view, detector="eyes")))
    assert clean.view.detector == "body"
    assert sanitize(AppSettings(view=replace(AppSettings().view, detector="both"))).view.detector == "both"


def test_sanitize_leaves_good_settings_alone():
    assert sanitize(AppSettings()) == AppSettings()


def test_settings_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        AppSettings().layout.rows = 9


def test_replace_path_changes_one_nested_field_only():
    from ren.settings import get_path, replace_path
    base = AppSettings()
    changed = replace_path(base, ("decision", "weights", "hot"), 0.8)
    assert changed.decision.weights.hot == 0.8 and changed.decision.weights.cold == base.decision.weights.cold
    assert changed.layout == base.layout and base.decision.weights.hot != 0.8
    assert get_path(changed, ("decision", "weights", "hot")) == 0.8


def test_replace_path_on_a_top_level_field():
    from ren.settings import replace_path
    assert replace_path(AppSettings(), ("view", "mirror"), True).view.mirror is True



# ---- venues -------------------------------------------------------------------------

from ren.settings import add_venue, remove_venue, select_venue, replace_path as _replace_path  # noqa: E402
from ren.venues import clean_name, suggest_name  # noqa: E402


def test_room_edits_land_in_the_active_venue_only():
    two = add_venue(AppSettings(), "lt2")
    edited = _replace_path(two, ("layout", "rows"), 9)
    assert edited.layout.rows == 9 and select_venue(edited, "lt1").layout.rows == PlanLayout().rows


def test_a_new_venue_copies_the_one_in_use_and_becomes_active():
    custom = _replace_path(AppSettings(), ("calibration",), Calibration((0.2, 0.2), (0.8, 0.2), (0.9, 0.9), (0.1, 0.9)))
    added = add_venue(custom, "lt2")
    assert added.venue == "lt2" and added.calibration == custom.calibration
    assert added.venue_names() == ["lt1", "lt2"]


def test_venue_names_are_cleaned_and_must_be_unique():
    assert clean_name("  LT   3 ", ["lt1"]) == "lt 3"
    for bad in ("", "   ", "lt1", "x" * 40):
        with pytest.raises(ValueError):
            clean_name(bad, ["lt1"])


def test_suggested_names_fill_the_next_gap():
    assert suggest_name(["lt1"]) == "lt2" and suggest_name(["lt1", "lt2", "lt4"]) == "lt3"


def test_removing_the_active_venue_falls_back_to_another():
    two = add_venue(AppSettings(), "lt2")
    left = remove_venue(two, "lt2")
    assert left.venue == "lt1" and left.venue_names() == ["lt1"]


def test_the_last_venue_cannot_be_removed():
    with pytest.raises(ValueError):
        remove_venue(AppSettings(), "lt1")


def test_unknown_venues_fail_loudly():
    with pytest.raises(KeyError):
        select_venue(AppSettings(), "nope")
    with pytest.raises(KeyError):
        remove_venue(add_venue(AppSettings(), "lt2"), "nope")


def test_venues_round_trip_through_disk(tmp_path):
    path = tmp_path / "s.json"
    settings = _replace_path(add_venue(AppSettings(), "lt2"), ("vent", "drop_m"), 3.5)
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded == settings and loaded.vent.drop_m == 3.5


def test_old_single_room_files_become_venue_lt1(tmp_path, caplog):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"layout": {"rows": 4, "cols": 8}, "impact": {"baseline_kw": 2.0}}))
    with caplog.at_level(logging.INFO, logger="ren.settings"):
        loaded = load_settings(path)
    assert loaded.venue_names() == ["lt1"] and loaded.layout == PlanLayout(4, 8)
    assert "unknown keys" not in caplog.text  # impact is dropped quietly


def test_sanitize_repairs_duplicate_names_and_a_missing_active_venue(caplog):
    messy = AppSettings(venues=(Venue("lt1"), Venue("lt1"), Venue("  ")), venue="gone")
    with caplog.at_level(logging.WARNING, logger="ren.settings"):
        clean = sanitize(messy)
    assert clean.venue_names() == ["lt1"] and clean.venue == "lt1"


def test_settings_with_no_venues_get_a_default_one():
    assert sanitize(AppSettings(venues=())).venue_names() == ["lt1"]


def test_removed_seats_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings(venues=(Venue("lt1", PlanLayout(3, 6, ((0, 2), (2, 5)))),))
    save_settings(settings, path)
    assert load_settings(path).layout.off == ((0, 2), (2, 5))


def test_removed_seats_outside_the_grid_are_dropped():
    settings = AppSettings(venues=(Venue("lt1", PlanLayout(2, 2, ((0, 1), (5, 5)))),))
    assert sanitize(settings).layout.off == ((0, 1),)
