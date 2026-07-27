import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import typer

from NPET_DP.framework.config import config

_COLUMN_GAP: int = 2
_DATETIME_PATTERN: re.Pattern[str] = re.compile(r"\d{8}_\d{6}")


def __get_data_files(ignored_files: Iterable[Path]) -> tuple[Path, ...]:
    """
    Get all the data files in the data directory.
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
    """
    Replace any `YYYYMMDD_HHMMSS` timestamp in the stem with `dd.mm.yyyy hh:mm:ss`,
    then turn remaining underscores into spaces.
    :param stem: File stem to format.
    :return: Formatted stem.
    """

    def format_match(match: re.Match[str]) -> str:
        return datetime.strptime(match.group(), "%Y%m%d_%H%M%S").strftime(
            "%d.%m.%Y %H:%M:%S"
        )

    return _DATETIME_PATTERN.sub(format_match, stem).replace("_", " ")


def __print_file_options(files: tuple[Path, ...]) -> None:
    """
    Print the numbered file options in as many columns as fit the terminal width,
    filling columns top-to-bottom before moving to the next column (like `ls`).
    :param files: Files to list.
    """
    index_width: int = len(str(len(files)))
    entries: list[str] = [
        f"{i:>{index_width}}: {__format_stem(file.stem)}"
        for i, file in enumerate(files, 1)
    ]
    entry_width: int = max(len(entry) for entry in entries) + _COLUMN_GAP
    terminal_width: int = shutil.get_terminal_size(fallback=(80, 24)).columns
    num_columns: int = max(1, min(len(entries), terminal_width // entry_width))
    num_rows: int = -(-len(entries) // num_columns)  # Ceiling division

    for row in range(num_rows):
        line_parts: list[str] = []
        for col in range(num_columns):
            index = col * num_rows + row
            if index < len(entries):
                is_last_column = col == num_columns - 1
                line_parts.append(
                    entries[index]
                    if is_last_column
                    else entries[index].ljust(entry_width)
                )
        typer.echo("".join(line_parts))


def user_file_select(
    file_description: str = "file",
    ignored_files: Iterable[Path] = (),
) -> Path:
    """
    Prompt the user to choose a file from the directory of data sources.
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
    typer.echo(f"Select {file_desc} from:")
    __print_file_options(files)
    while True:
        # Subtract 1 to make the index 0-based
        choice: int = typer.prompt("Insert number of your selection", type=int) - 1
        try:
            return files[choice]
        except IndexError:
            typer.secho("Invalid choice!", fg=typer.colors.RED)
