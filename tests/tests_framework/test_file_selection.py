from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from NPET_DP.framework.config import config
from NPET_DP.framework.file_selection import user_file_select


@pytest.fixture()
def _test_files(tmp_path: Path) -> tuple[Path, Path]:
    """Fixture to create multiple .out files for testing."""
    files: list[Path] = [tmp_path / "first.out", tmp_path / "second.out"]
    for f in files:
        f.write_text("dummy", encoding="utf-8")
    files = sorted(files, key=lambda p: p.name)
    files_t: tuple[Path, Path] = cast(tuple[Path, Path], tuple(files))
    return files_t


def test_user_file_select_raises_when_no_files_and_user_quits(tmp_path, monkeypatch):
    """Test that user_file_select raises when no files are found and the user quits."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.confirm", return_value=False):
        with pytest.raises(FileNotFoundError):
            user_file_select()


def test_user_file_select_returns_sole_out_file(tmp_path, monkeypatch):
    """Test that user_file_select returns the sole .out file when there is only one."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    file_path = tmp_path / "single.out"
    file_path.write_text("dummy", encoding="utf-8")
    assert user_file_select() == file_path


def test_user_file_select_chooses_from_multiple_files(
    tmp_path,
    monkeypatch,
    _test_files,
):
    """Test that user_file_select chooses from multiple out files."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", return_value=1 + 1):  # 1-based index
        result: Path = user_file_select()
    assert result == _test_files[1], (
        f"Result: {result}, expected: {_test_files[1]}, Possible choices: {_test_files}"
    )


def test_user_file_select_reprompts_on_invalid_choice(
    tmp_path, monkeypatch, _test_files
):
    """Test that user_file_select reprompts on invalid choice."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", side_effect=[99, 1 + 1]):  # 1-based index
        result: Path = user_file_select()
    assert result == _test_files[1], (
        f"Result: {result}, expected: {_test_files[1]}, Possible choices: {_test_files}"
    )


def test_user_file_select_ignores_non_out_files(tmp_path, monkeypatch):
    """Test that user_file_select ignores non-out files."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    (tmp_path / "ignore.txt").write_text("dummy", encoding="utf-8")
    expected = tmp_path / "valid.out"
    expected.write_text("dummy", encoding="utf-8")
    result: Path = user_file_select()
    assert result == expected


def test_user_file_select_ignore_files(
    _test_files,
    tmp_path,
    monkeypatch,
):
    """Test that user_file_select can ignore specific files by name."""
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    result: Path = user_file_select(ignored_files=[_test_files[0]])
    assert result == _test_files[1]
