import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import typer

from NPET_DP.framework.config import config

_COLUMN_GAP: int = 2
_ROWS_PER_PAGE: int = 5
_DATETIME_PATTERN: re.Pattern[str] = re.compile(r"\d{8}_\d{6}")


def __get_data_files(ignored_files: Iterable[Path]) -> tuple[Path, ...]:
    """Get all the data files in the data directory.

    :param ignored_files: Files to ignore.
    :return: Tuple of data files.
    """
    return tuple(
        sorted(
            (f for f in config.input_data_dir.glob("*.out") if f not in ignored_files),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )


def __format_stem(stem: str) -> str:
    """Format datetime in the filename.

    Replace any `YYYYMMDD_HHMMSS` timestamp in the stem with `dd.mm.yyyy hh:mm:ss`,
    then turn remaining underscores into spaces.
    :param stem: File stem to format.
    :return: Formatted stem.
    """

    def format_match(match: re.Match[str]) -> str:
        return (
            datetime.strptime(match.group(), "%Y%m%d_%H%M%S")
            .replace(tzinfo=UTC)
            .strftime("%d.%m.%Y %H:%M:%S")
        )

    return _DATETIME_PATTERN.sub(format_match, stem).replace("_", " ")


def __build_entries(files: tuple[Path, ...]) -> tuple[list[str], int, int]:
    """Build the numbered entry labels and figure out how many columns fit the terminal width.

    :param files: Files to list.
    :return: Tuple of (entries, entry width including a gap, number of columns).
    """
    index_width: int = len(str(len(files)))
    entries: list[str] = [
        f"{i:>{index_width}}: {__format_stem(file.stem)}"
        for i, file in enumerate(files, 1)
    ]
    entry_width: int = max(len(entry) for entry in entries) + _COLUMN_GAP
    terminal_width: int = shutil.get_terminal_size(fallback=(80, 24)).columns
    num_columns: int = max(1, min(len(entries), terminal_width // entry_width))
    return entries, entry_width, num_columns


def __print_file_options_page(
    entries: list[str],
    entry_width: int,
    num_columns: int,
    page: int,
) -> None:
    """Print one page of file selection.

    The printed page (at most `_ROWS_PER_PAGE` rows) has the numbered file options,
    filling columns top-to-bottom before moving to the next column (like `ls`).
    :param entries: All numbered entry labels.
    :param entry_width: Column width including a gap.
    :param num_columns: Number of columns to lay entries out in.
    :param page: Zero-based page index to print.
    """
    page_size: int = num_columns * _ROWS_PER_PAGE
    page_entries: list[str] = entries[page * page_size : (page + 1) * page_size]
    num_rows: int = -(-len(page_entries) // num_columns)  # Ceiling division
    for row in range(num_rows):
        line_parts: list[str] = []
        for col in range(num_columns):
            index = col * num_rows + row
            if index < len(page_entries):
                is_last_column = col == num_columns - 1
                line_parts.append(
                    page_entries[index]
                    if is_last_column
                    else page_entries[index].ljust(entry_width)
                )
        typer.echo("".join(line_parts))


def user_file_select(
    file_description: str = "file",
    ignored_files: Iterable[Path] = (),
) -> Path:
    """Prompt the user to choose a file from the directory of data sources.

    :param file_description: Description of the file that will be used in the prompt.
    :param ignored_files: Files to ignore.
    :return: Path of the chosen file.
    :raises FileNotFoundError: If no files are found in the specified directory.
    """
    file_desc: str = file_description.lower().strip()
    files: tuple[Path, ...] = __get_data_files(ignored_files)
    while not files:
        # If there are no files found, prompt the user to insert them in the correct dir
        typer.secho("\nNo valid data found!", fg=typer.colors.RED)
        typer.echo(f"Either insert the data to process here: {config.input_data_dir}")
        typer.echo("Or change the data directory in settings?")
        if not typer.confirm("Quit to main menu (N) or continue (Y)?", default=True):
            raise FileNotFoundError
        files = __get_data_files(ignored_files)
    # If only a single file is available, automatically select it
    if len(files) == 1:
        typer.echo(f"Automatically selected sole {file_desc} file: {files[0]}")
        return files[0]
    # Otherwise let the user choose from the available files
    entries, entry_width, num_columns = __build_entries(files)
    page_size: int = num_columns * _ROWS_PER_PAGE
    total_pages: int = -(-len(entries) // page_size)  # Ceiling division
    page: int = 0
    prompt_text: str = "Insert number of your selection (0: enter filename manually)"
    if total_pages > 1:
        prompt_text += " (-1: previous page, -2: next page)"
    typer.echo(f"Select {file_desc} from:")
    __print_file_options_page(entries, entry_width, num_columns, page)
    while True:
        choice: int = typer.prompt(prompt_text, type=int)
        if choice == -1:
            if page > 0:
                page -= 1
            __print_file_options_page(entries, entry_width, num_columns, page)
            continue
        if choice == -2:
            if page < total_pages - 1:
                page += 1
            __print_file_options_page(entries, entry_width, num_columns, page)
            continue
        if choice == 0:
            manual_path: Path = Path(typer.prompt("Enter file path")).expanduser()
            if not manual_path.is_absolute():
                manual_path = config.input_data_dir / manual_path
            if not manual_path.suffix:
                manual_path = manual_path.with_suffix(".out")
            if manual_path.is_file():
                return manual_path
            typer.secho(f"File not found: {manual_path}", fg=typer.colors.RED)
            continue
        if choice < 0:
            typer.secho("Invalid choice!", fg=typer.colors.RED)
            continue
        try:
            # Subtract 1 to make the index 0-based
            return files[choice - 1]
        except IndexError:
            typer.secho("Invalid choice!", fg=typer.colors.RED)
    raise FileNotFoundError("Failed to select file")
