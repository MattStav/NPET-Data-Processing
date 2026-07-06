import os
import sys
from pathlib import Path

import typer

from NPET_DP import __file__ as app_file
from NPET_DP.framework.constants import APPDATA_DIR_NAME


def __is_dev_env() -> bool:
    """
    Check if the app is running in a development environment.
    :return: True if the app is running in a development environment, False otherwise.
    """
    return __app_file_path().parent.parent.name == "src"


def __app_file_path() -> Path:
    """Get the path to the app __main__ file."""
    # noinspection PyTypeChecker
    return Path(app_file)


def get_path(append: str = "") -> Path:
    """
    Get a path to the directory where data is stored.
    :param append: Path to append to the base path.
    """
    base_path: Path = (
        __app_file_path().parents[2] / APPDATA_DIR_NAME
        if __is_dev_env()
        else Path(os.environ["APPDATA"]) / APPDATA_DIR_NAME
    )
    base_path.mkdir(exist_ok=True)
    return base_path / append


def get_plot_path(file_name: str = "", *, suffix: str = ".png") -> Path:
    """
    Get a path to the directory where plots are stored.
    :param file_name: Name of the plot file.
    :param suffix: Suffix of the plot file. It is appended to the file name if not already present.
    :return: Path to the plot file.
    """
    if not suffix:
        raise ValueError("Suffix cannot be empty. Please provide a valid suffix.")
    if file_name and suffix and not file_name.endswith(suffix):
        if not suffix.startswith("."):
            suffix = "." + suffix
        file_name += suffix
    plots_path: Path = get_path("DP_plots")
    plots_path.mkdir(exist_ok=True)
    return plots_path / file_name


def open_plot_outputs() -> None:
    """Open the directory where plots are stored."""
    outputs_dir: Path = get_plot_path()
    if sys.platform == "win32":
        os.startfile(outputs_dir)
    else:
        typer.secho("Not supported on this platform", fg=typer.colors.RED)
        typer.echo(f"Please open the outputs manually here: {outputs_dir}")
