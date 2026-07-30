from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from NPET_DP.framework.constants import APPDATA_DIR_NAME
from NPET_DP.framework.path_handler import get_path, get_plot_path, open_plot_outputs


def _fake_app_file(
    tmp_path: Path, monkeypatch: MonkeyPatch, parent_dir_name: str
) -> Path:
    """Create a fake ``NPET_DP/__init__.py`` and point ``path_handler.app_file`` at it.

    ``parent_dir_name`` decides which branch ``get_path`` takes: "src" makes the file look
    like it's running from a development checkout, anything else (e.g. "site-packages")
    makes it look like an installed package.

    :param tmp_path: Root directory to build the fake package layout under.
    :param monkeypatch: Used to patch ``path_handler.app_file`` to the fake file's path.
    :param parent_dir_name: Name of the directory that contains the "NPET_DP" package dir.
    :return: Path to the fake ``__init__.py`` file.
    """
    app_file = tmp_path / parent_dir_name / "NPET_DP" / "__init__.py"
    app_file.parent.mkdir(parents=True)
    app_file.write_text("", encoding="utf-8")
    monkeypatch.setattr("NPET_DP.framework.path_handler.app_file", str(app_file))
    return app_file


@pytest.fixture(autouse=True)
def _non_dev_app_file(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Default every test to a non-development app file location.

    Without this, tests targeting the APPDATA branch would behave differently depending on
    whether the suite itself runs from a "src" checkout or an installed package.
    """
    _fake_app_file(tmp_path, monkeypatch, "site-packages")


def test_get_path_dev_env(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test dev env pathing.

    Test that get_path resolves relative to the app file's great-grandparent directory
    when running in a development environment (the app file's grandparent dir is named "src").
    """
    _fake_app_file(tmp_path, monkeypatch, "src")
    p = get_path("test.txt")
    assert isinstance(p, Path)
    assert p.name == "test.txt"
    assert p == tmp_path / APPDATA_DIR_NAME / "test.txt"


def test_get_path_appdata(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that get_path returns a Path object and resolves relative to APPDATA.

    Test that get_path returns a Path object and resolves relative to APPDATA,
    when not running in a development environment.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    p = get_path("test.txt")
    assert isinstance(p, Path)
    assert p.name == "test.txt"
    assert p.parent.is_dir(), "The parent directory should be created"
    assert p == tmp_path / APPDATA_DIR_NAME / "test.txt"


def test_get_plot_path(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that get_plot_path returns a directory nested under get_path."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path = get_plot_path()
    assert plot_path.is_dir()
    assert plot_path.is_relative_to(get_path())


def test_get_plot_path_leads_to_dp_plots_dir(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that get_plot_path resolves into a "DP_plots" subdirectory."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path = get_plot_path()
    assert plot_path.name == "DP_plots"
    assert plot_path.name.startswith("DP_"), "The name should start with `DP`"
    assert plot_path == tmp_path / APPDATA_DIR_NAME / "DP_plots"


def test_get_plot_path_appends_suffix_default(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that the default suffix is .png when not specified."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path = get_plot_path("test")
    assert plot_path.suffix == ".png"


@pytest.mark.parametrize("suffix", [".png", ".html", "png"])
def test_get_plot_path_appends_suffix(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    suffix: str,
) -> None:
    """Test that the suffix is correctly appended to the file name.

    The suffix should be correctly appended even when the dot is not specified.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    plot_path = get_plot_path("test", suffix=suffix)
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    assert plot_path.suffix == suffix


def test_get_plot_path_empty_suffix_raises(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that an empty suffix raises a ValueError."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    with pytest.raises(ValueError, match="Suffix cannot be empty"):
        get_plot_path("test", suffix="")


@patch("NPET_DP.framework.path_handler.sys.platform", "win32")
@patch("NPET_DP.framework.path_handler.os.startfile", create=True)
def test_open_plot_outputs_windows(
    mock_startfile: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that open_plot_outputs opens the plot dir via os.startfile on Windows."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    open_plot_outputs()
    mock_startfile.assert_called_once_with(get_plot_path())


@patch("NPET_DP.framework.path_handler.sys.platform", "linux")
def test_open_plot_outputs_non_windows(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that open_plot_outputs prints a manual instruction on non-Windows platforms."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    open_plot_outputs()
    captured = capsys.readouterr()
    assert "Not supported on this platform" in captured.out
    assert str(get_plot_path()) in captured.out
