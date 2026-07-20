from pathlib import Path

import numpy as np
import numpy.typing as npt
import typer
from matplotlib import pyplot as plt

from NPET_DP.processing.helpers import import_data, validate_inputs
from NPET_DP.framework.config import config
from NPET_DP.framework.constants import FEMTO
from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path


@validate_inputs
def __plot_singular_data(data: npt.NDArray, name: str) -> None:
    """
    Plot a time epoch of each measurement within a set bin of width 1/freq
    Showing the delay within each measurement time window.
    WARNING: The yaxis values are nonsense
    :param data: Data to be plotted, in the FW standard format.
    :param name: Name of the file
    """
    typer.echo(f"Plotting measured data from {name}")
    # This "wraps" each timestamp into a time window of width 1/freq (one measurement period)
    # This shows where within each frequency cycle the measurement occurred
    plot_data = np.mod(data["femto"], FEMTO / config.frequency)
    plt.scatter([*range(len(plot_data))], plot_data, s=0.01)
    plt.title("Modulo: Epoch / Measurement window")
    plt.savefig(get_plot_path(f"single_{name}"))
    plt.show(block=False)


def main_one_epoch() -> None:
    """Plot singular epoch data."""
    try:
        epoch_file_path: Path = user_file_select()
    except FileNotFoundError:
        return
    __plot_singular_data(import_data(epoch_file_path), epoch_file_path.stem)
