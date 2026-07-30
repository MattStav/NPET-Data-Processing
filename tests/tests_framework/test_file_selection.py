import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import CaptureFixture, MonkeyPatch

from NPET_DP.framework.config import config
from NPET_DP.framework.file_selection import _ROWS_PER_PAGE, user_file_select


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


@pytest.fixture()
def _force_single_column(monkeypatch: MonkeyPatch) -> None:
    """Force the file-selection layout to a single column.

    This makes `page_size` (`num_columns * _ROWS_PER_PAGE`) deterministic and
    equal to `_ROWS_PER_PAGE`, regardless of the terminal width the tests run in.
    """
    monkeypatch.setattr(
        "NPET_DP.framework.file_selection.shutil.get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((1, 24)),
    )


@pytest.fixture()
def _paginated_files(tmp_path: Path) -> tuple[Path, ...]:
    """Create one more file than fits on a single page (with a single column forced).

    Returned newest-first, matching the sort order used by user_file_select.
    """
    now: float = time.time()
    paths: list[Path] = []
    for i in range(_ROWS_PER_PAGE + 2):
        f = tmp_path / f"file{i}.out"
        f.write_text("dummy", encoding="utf-8")
        os.utime(f, (now + i, now + i))
        paths.append(f)
    return tuple(reversed(paths))


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


def test_user_file_select_manual_path_absolute(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
) -> None:
    """Test manual path entry with an absolute path.

    Choosing 0 should prompt for a file path and, when given an absolute
    path to an existing file, return it directly.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    other_dir: Path = tmp_path / "elsewhere"
    other_dir.mkdir()
    target: Path = other_dir / "manual.out"
    target.write_text("dummy", encoding="utf-8")
    with patch("typer.prompt", side_effect=[0, str(target)]):
        result: Path = user_file_select()
    assert result == target


def test_user_file_select_manual_path_relative_to_data_dir(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
) -> None:
    """Test manual path entry with a relative path.

    A relative path should be resolved against the configured data directory.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    target: Path = tmp_path / "manual.out"
    target.write_text("dummy", encoding="utf-8")
    with patch("typer.prompt", side_effect=[0, "manual.out"]):
        result: Path = user_file_select()
    assert result == target


def test_user_file_select_manual_path_without_suffix_appends_out(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
) -> None:
    """Test manual path entry without a suffix.

    When the entered path has no suffix, `.out` should be appended before
    looking up the file.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    target: Path = tmp_path / "manual.out"
    target.write_text("dummy", encoding="utf-8")
    with patch("typer.prompt", side_effect=[0, "manual"]):
        result: Path = user_file_select()
    assert result == target


def test_user_file_select_manual_path_not_found_reprompts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _test_files: tuple[Path, Path],
    capsys: CaptureFixture,
) -> None:
    """Test manual path entry with a nonexistent file.

    Entering a path that doesn't resolve to an existing file should print an
    error and return to the main selection prompt instead of raising.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", side_effect=[0, "missing.out", 1]):
        result: Path = user_file_select()
    assert result == _test_files[0]
    assert "File not found" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    ("num_files", "expect_hint"),
    [(_ROWS_PER_PAGE, False), (_ROWS_PER_PAGE + 1, True)],
)
def test_user_file_select_pagination_hint_boundary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _force_single_column: None,
    num_files: int,
    expect_hint: bool,
) -> None:
    """Test that the pagination hint only appears once files overflow a single page.

    With a single column forced, `page_size` equals `_ROWS_PER_PAGE`, so exactly
    `_ROWS_PER_PAGE` files should still fit on one page (no hint), while one more
    file should require a second page (hint shown).
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    now: float = time.time()
    for i in range(num_files):
        f = tmp_path / f"file{i}.out"
        f.write_text("dummy", encoding="utf-8")
        os.utime(f, (now + i, now + i))
    with patch("typer.prompt", return_value=1) as mock_prompt:
        user_file_select()
    prompt_text: str = mock_prompt.call_args_list[0].args[0]
    assert ("next page" in prompt_text) == expect_hint
    assert ("previous page" in prompt_text) == expect_hint


def test_user_file_select_previous_page_navigates_back(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _paginated_files: tuple[Path, ...],
    _force_single_column: None,
) -> None:
    """Test that choosing -1 after -2 returns to the first page.

    Selecting index 1 after paging forward and back should resolve to the
    newest file, proving -1 actually paged backward.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", side_effect=[-2, -1, 1]):
        result: Path = user_file_select()
    assert result == _paginated_files[0]


def test_user_file_select_next_page_clamps_at_last_page(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _paginated_files: tuple[Path, ...],
    _force_single_column: None,
) -> None:
    """Test that -2 beyond the last page keeps the current page instead of erroring.

    Repeated -2 choices past the final page should not raise or skip past valid
    entries; the last entry should still be selectable afterward.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    last_index: int = len(_paginated_files)
    with patch("typer.prompt", side_effect=[-2, -2, last_index]):
        result: Path = user_file_select()
    assert result == _paginated_files[last_index - 1]


def test_user_file_select_previous_page_clamps_at_first_page(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _paginated_files: tuple[Path, ...],
    _force_single_column: None,
) -> None:
    """Test that -1 on the first page keeps the current page instead of erroring.

    Choosing -1 while already on the first page should not raise or move to a
    negative page; the newest entry should still be selectable afterward.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    with patch("typer.prompt", side_effect=[-1, 1]):
        result: Path = user_file_select()
    assert result == _paginated_files[0]


def test_user_file_select_page_shows_only_current_page_entries(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    _paginated_files: tuple[Path, ...],
    _force_single_column: None,
    capsys: CaptureFixture,
) -> None:
    """Test that each page only prints its own slice of entries.

    Entries belonging to the other page should not be visible on screen at the
    same time as the currently displayed page.
    """
    monkeypatch.setattr(config, "input_data_dir", tmp_path)
    choices: list[int] = [-2, len(_paginated_files)]

    def fake_prompt(*_args: object, **_kwargs: object) -> int:
        output: str = capsys.readouterr().out
        if len(choices) == 2:
            for i in range(1, _ROWS_PER_PAGE + 1):
                assert f"{i}: " in output
            assert f"{_ROWS_PER_PAGE + 1}: " not in output
            assert f"{_ROWS_PER_PAGE + 2}: " not in output
        else:  # second call, second page just printed
            assert f"{_ROWS_PER_PAGE + 1}: " in output
            assert f"{_ROWS_PER_PAGE + 2}: " in output
            for i in range(1, _ROWS_PER_PAGE + 1):
                assert f"{i}: " not in output
        return choices.pop(0)

    with patch("typer.prompt", side_effect=fake_prompt):
        result: Path = user_file_select()
    assert result == _paginated_files[-1]
