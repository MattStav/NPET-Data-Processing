from pathlib import Path
from unittest.mock import patch

import pytest

from NPET_DP.framework.path_handler import get_path, get_plot_path, open_plot_outputs


@pytest.fixture(autouse=True)
def _non_dev_app_file(tmp_path, monkeypatch):
    """
    Default to a non-development app file location, so tests targeting the
    APPDATA branch aren't affected by whether the suite itself runs from a "src" checkout.
    """
    fake_app_file = tmp_path / "site-packages" / "NPET_DP" / "__init__.py"
    fake_app_file.parent.mkdir(parents=True)
    fake_app_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("NPET_DP.framework.path_handler.app_file", str(fake_app_file))


def test_get_path_dev_env(tmp_path, monkeypatch):
    """
    Test that get_path resolves relative to the app file's great-grandparent directory
    when running in a development environment (the app file's grandparent dir is named "src").
    """
    fake_app_file = tmp_path / "src" / "NPET_DP" / "__init__.py"
    fake_app_file.parent.mkdir(parents=True)
    fake_app_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("NPET_DP.framework.path_handler.app_file", str(fake_app_file))
    p: Path = get_path("test.txt")
    assert isinstance(p, Path)
    assert p.name == "test.txt"
    assert p == tmp_path / "NPET" / "test.txt"


def test_get_path_appdata(tmp_path, monkeypatch):
    """
    Test that get_path returns a Path object and resolves relative to APPDATA,
    when not running in a development environment.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    p = get_path("test.txt")
    assert isinstance(p, Path)
    assert p.name == "test.txt"
    assert p.parent.is_dir(), "The parent directory should be created"
    assert p == tmp_path / "NPET" / "test.txt"


def test_get_plot_path(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path: Path = get_plot_path()
    assert plot_path.is_dir()
    assert plot_path.is_relative_to(get_path())


def test_get_plot_path_leads_to_dp_plots_dir(tmp_path, monkeypatch):
    """Test that get_plot_path resolves into a "DP_plots" subdirectory."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path: Path = get_plot_path()
    assert plot_path.name == "DP_plots"
    assert plot_path == tmp_path / "NPET" / "DP_plots"


def test_get_plot_path_appends_suffix_default(tmp_path, monkeypatch):
    """Test that the default suffix is .png when not specified."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path: Path = get_plot_path("test")
    assert plot_path.suffix == ".png"


@pytest.mark.parametrize("suffix", [".png", ".html", "png"])
def test_get_plot_path_appends_suffix(tmp_path, monkeypatch, suffix: str):
    """Test that the suffix is correctly appended to the file name."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path: Path = get_plot_path("test", suffix=suffix)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    assert plot_path.suffix == suffix


def test_get_plot_path_empty_suffix_raises(tmp_path, monkeypatch):
    """Test that an empty suffix raises a ValueError."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    with pytest.raises(ValueError):
        get_plot_path("test", suffix="")


@patch("NPET_DP.framework.path_handler.sys.platform", "win32")
@patch("NPET_DP.framework.path_handler.os.startfile", create=True)
def test_open_plot_outputs_windows(mock_startfile, tmp_path, monkeypatch):
    """Test that open_plot_outputs opens the plot dir via os.startfile on Windows."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    open_plot_outputs()
    mock_startfile.assert_called_once_with(get_plot_path())


@patch("NPET_DP.framework.path_handler.sys.platform", "linux")
def test_open_plot_outputs_non_windows(tmp_path, monkeypatch, capsys):
    """Test that open_plot_outputs prints a manual instruction on non-Windows platforms."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    open_plot_outputs()
    captured = capsys.readouterr()
    assert "Not supported on this platform" in captured.out
    assert str(get_plot_path()) in captured.out
