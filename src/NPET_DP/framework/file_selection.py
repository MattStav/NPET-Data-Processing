from pathlib import Path
from typing import Iterable

import typer

from NPET_DP.framework.config import config


def __get_data_files(ignored_files: Iterable[Path]) -> tuple[Path, ...]:
    """
    Get all the data files in the data directory.
    :param ignored_files: Files to ignore.
    :return: Tuple of data files.
    """
    return tuple(
        sorted(
            (f for f in config.input_data_dir.glob("*.out") if f not in ignored_files),
            key=lambda p: p.stem,
        )
    )


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
    for i, file in enumerate(files, 1):
        typer.echo(f"\t{i}: {file.stem}")
    while True:
        # Subtract 1 to make the index 0-based
        choice: int = typer.prompt("Insert number of your selection", type=int) - 1
        try:
            return files[choice]
        except IndexError:
            typer.secho("Invalid choice!", fg=typer.colors.RED)
