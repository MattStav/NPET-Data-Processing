import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture, MonkeyPatch

from NPET_DP.framework.config import config
from NPET_DP.framework.file_selection import user_file_select


@pytest.fixture()
def _test_files(tmp_path: Path) -> tuple[Path, Path]:
    """Fixture to create two .out files.

    The files have distinct mtimes, returned newest-first
    to match the newest-to-oldest sort order used by user_file_select.
    """
    older, newer = tmp_path / "first.out", tmp_path / "second.out"
    for f in (older, newer):
        f.write_text("dummy", encoding="utf-8")
    now: float = time.time()
    os.utime(older, (now - 60, now - 60))
    os.utime(newer, (now, now))
    return newer, older


def test_user_file_select_raises(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test that user_file_select raise.

    It should raise when no files are found and the user quits.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.confirm", return_value=False):  # noqa: SIM117
        with pytest.raises(FileNotFoundError):
            user_file_select()


def test_user_file_select_sole_out_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Test that user_file_select on sole file.

    When there is only a single .out file, user_file_select should automatically select it.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    file_path = tmp_path / "single.out"
    file_path.write_text("dummy", encoding="utf-8")
    assert user_file_select() == file_path


def test_user_file_select_multiple_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
) -> None:
    """Test that user_file_select chooses from multiple out files.

    When there are several files, the user should be presented the files
    and then be prompted to choose one via a number representation.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", return_value=1 + 1):  # 1-based index
        result: Path = user_file_select()
    assert result == _test_files[1], (
        f"Result: {result}, expected: {_test_files[1]}, Possible choices: {_test_files}"
    )


def test_user_file_select_reprompts_on_invalid_choice(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
) -> None:
    """Test that user_file_select reprompts on invalid choice.

    The user should be reprompted until a valid selection is made.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", side_effect=[99, 1 + 1]):  # 1-based index
        result: Path = user_file_select()
    assert result == _test_files[1], (
        f"Result: {result}, expected: {_test_files[1]}, Possible choices: {_test_files}"
    )


def test_user_file_select_ignores_non_out_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that user_file_select ignores non-out files.

    All measured data should be in .out files, there is no need to process any other files.
    This tests presents only one valid files, which should be directly selected.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    (tmp_path / "ignore.txt").write_text("dummy", encoding="utf-8")
    expected: Path = tmp_path / "valid.out"
    expected.write_text("dummy", encoding="utf-8")
    result: Path = user_file_select()
    assert result == expected


def test_user_file_select_ignore_files(
    _test_files: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that user_file_select can ignore specific files by name.

    When provided with an Iterable with Paths the function should ignore the Paths therein.
    This is useful when selecting multiple files in a row, the selected ones are excluded from the next selection.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    result: Path = user_file_select(ignored_files=[_test_files[0]])
    assert result == _test_files[1]


def test_user_file_select_formats_embedded_timestamp(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    """Test timestamp formatting.

    Test that a YYYYMMDD_HHMMSS timestamp in a file name is displayed as dd.mm.yyyy hh:mm:ss.
    This increases selection readability.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    (tmp_path / "scan_20260723_134917.out").write_text("dummy", encoding="utf-8")
    (tmp_path / "scan_20260101_080000.out").write_text("dummy", encoding="utf-8")
    with patch("typer.prompt", return_value=1):
        user_file_select()
    output = capsys.readouterr().out
    assert "23.07.2026 13:49:17" in output
    assert "01.01.2026 08:00:00" in output
    assert "20260723_134917" not in output
    assert "20260101_080000" not in output
