from pathlib import Path

import numpy as np
import typer
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from NPET_DP.epoch_processing.helper_funcs import (
    auto_scale_data,
    get_unit,
    import_data,
    validate_inputs,
)
from NPET_DP.epoch_processing.helper_plot import plot_time_deviation
from NPET_DP.epoch_processing.helper_processing import is_continuous, process_overflow
from NPET_DP.epoch_processing.helper_processing_prompted import drift_removal_prompt
from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path


def __plot_short(data: np.ndarray, name: str, y_units: str) -> None:
    """
    Show the data using a scatter plot.
    :param data: Data to be plotted, the first column is seconds, the second column is delay.
    :param name: Name of the file.
    :param y_units: Units of the delay data (y-axis).
    """
    typer.echo("Plotting PPS data...")
    plt.plot(
        range(len(data)),
        data[:, 1],
        marker="o",
        linestyle="-",
        linewidth=1,
        markersize=5,
    )
    plt.ylabel(f"NPET clock to PPS difference [{y_units}]")
    plt.xlabel("Measurement number [n]")
    plt.title("PPS Data Plot")
    plt.grid()
    plt.savefig(get_plot_path(f"pps_{name}"))
    plt.show(block=False)


def __plot_long(data: NDArray, name: str, y_units: str) -> None:
    """
    Plot a large amount of PPS data in a scatter plot.
    :param data: Data to be plotted, the first column is seconds, the second column is delay.
    :param name: Name of the file.
    :param y_units: Units of the delay data (y-axis).
    """
    typer.echo("Plotting large PPS dataset...")
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(
        x,
        y,
        marker="o",
        linestyle="None",
        markersize=5,
        alpha=0.5,
    )
    # Calculate moving average (window size is 5% of data, min 10 seconds)
    window_s: int = max(10, len(data) // 20)
    window: NDArray[np.float64] = np.asarray(
        np.ones(window_s, dtype=np.float64) / window_s,
        dtype=np.float64,
    )
    moving_avg = np.convolve(y, window, mode="valid")
    # Shift x for moving average to center it
    x_avg = x[window_s // 2 : -(window_s // 2) + (1 if window_s % 2 == 0 else 0)]
    plt.plot(
        x_avg,
        moving_avg,
        color="red",
        linewidth=2,
        label=f"Moving Avg (w = {window_s} s)",
    )
    plt.ylabel(f"NPET clock to PPS difference [{y_units}]")
    plt.xlabel("Measurement second [s]")
    plt.title(f"PPS Plot - {name}")
    plt.legend()
    plt.grid()
    plt.savefig(get_plot_path(f"pps_{name}"))
    plt.show(block=False)


@validate_inputs
def __plot_crossroad(data: NDArray, name: str) -> None:
    """
    Plot the PPS data, using a different plotting method depending on the size of the data.
    :param data: Data to be plotted, the first column is seconds, the second column is delay.
    :param name: Name of the file.
    """
    sc_femto, scale_num = auto_scale_data(data["femto"])
    y_units = get_unit("fs", scale_num)
    ave: float = float(np.average(sc_femto))
    typer.secho(f"{name} mean delay = {ave:.5f} {y_units}", bold=True)
    std: float = float(np.std(sc_femto))
    typer.echo(f"{name} STD = {std:.4f} {y_units}")
    full_data: NDArray[np.floating] = np.column_stack((data["seconds"], sc_femto))
    if len(data) > 600:
        __plot_long(full_data, name, y_units)
    else:
        __plot_short(full_data, name, y_units)


def main_pps() -> None:
    """Plot the PPS measurement. Used to evaluate the PPS stability."""
    try:
        pps_file_path: Path = user_file_select()
    except FileNotFoundError:
        return
    data_pps: NDArray = import_data(pps_file_path)
    processed: NDArray = process_overflow(data_pps)
    # Plot the data
    typer.echo()
    __plot_crossroad(processed, name=pps_file_path.stem)
    # Plot Allan time deviation
    if not is_continuous(processed["seconds"]):
        typer.secho("Data not continuous, skipping TDEV!", fg=typer.colors.RED)
    typer.echo()
    typer.echo("Plotting TDEV...")
    processed_drift_comp, deg = drift_removal_prompt(processed)
    plot_time_deviation(processed_drift_comp, 1, name=pps_file_path.stem)
    if deg:
        typer.echo("Plotting PPS data after drift removal...")
        __plot_crossroad(
            processed_drift_comp,
            name=f"{pps_file_path.stem} without pol deg {deg} drift",
        )
