from pathlib import Path

import numpy as np
import typer
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from NPET_DP.framework.config import config
from NPET_DP.framework.constants import FEMTO
from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import auto_range, histogram_plot_loop


def __plot_singular_data(data: NPETData, name: str) -> None:
    """
    Plot a time epoch of each measurement within a set bin of width 1/freq
    Showing the delay within each measurement time window.
    WARNING: The yaxis values are nonsense
    :param data: Data to be plotted, as NPETData object.
    :param name: Name of the file
    """
    typer.echo(f"Plotting measured data from {name}")
    sc_femto, unit = data.sc_femto
    plt.scatter([*range(len(sc_femto))], sc_femto, s=0.01)
    plt.title("Modulo: Epoch / Measurement window")
    plt.ylabel(f"Modulo delay [{unit}]")
    plt.xlabel("Measurement [n]")
    plt.savefig(get_plot_path(f"single_{name}"))
    plt.show(block=False)


def main_one_epoch() -> None:
    """Plot singular epoch data."""
    try:
        epoch_file_path: Path = user_file_select()
    except FileNotFoundError:
        return
    typer.echo(f"Importing data from {epoch_file_path}")
    data = NPETData.from_path(epoch_file_path)
    freq: int = config.frequency
    modulo_value: int = round(FEMTO / freq)
    typer.echo(f"Calculating Modulo with value = {modulo_value}")
    # This "wraps" each timestamp into a time window of width 1/freq (one measurement period)
    # This shows where within each frequency cycle the measurement occurred
    mod_data = data.modulo(modulo_value)
    __plot_singular_data(mod_data, epoch_file_path.stem)
    autodetection: tuple[NDArray[np.bool_], ...] = mod_data.detect_signal()
    if len(autodetection) == 1:
        typer.echo("Autodetection found a single signal")
        auto_range(mod_data, autodetection[0])
    else:
        typer.echo("Unable to autodetect a single signal")
    if len(autodetection) != 1 and not typer.confirm("Do you want to plot histogram?"):
        return
    histogram_plot_loop(mod_data, epoch_file_path.stem)
