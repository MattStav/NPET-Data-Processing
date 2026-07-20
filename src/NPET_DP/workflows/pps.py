from pathlib import Path

import numpy as np
import typer
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.plotting import plot_time_deviation
from NPET_DP.workflows.helpers import drift_removal_prompt


def __plot_short(data: NPETData, name: str) -> None:
    """
    Show the data using a scatter plot.
    :param data: Data to be plotted, as NPETData object.
    :param name: Name of the file.
    """
    typer.echo("Plotting PPS data...")
    sc_femto, unit = data.sc_femto
    plt.plot(
        range(len(sc_femto)),
        sc_femto,
        marker="o",
        linestyle="-",
        linewidth=1,
        markersize=5,
    )
    plt.ylabel(f"NPET clock to PPS difference [{unit}]")
    plt.xlabel("Measurement number [n]")
    plt.title("PPS Data Plot")
    plt.grid()
    plt.savefig(get_plot_path(f"pps_{name}"))
    plt.show(block=False)


def __plot_long(data: NPETData, name: str) -> None:
    """
    Plot a large amount of PPS data in a scatter plot.
    :param data: Data to be plotted, as NPETData object.
    :param name: Name of the file.
    """
    typer.echo("Plotting large PPS dataset...")
    sc_data, unit = data.sc_structured_arr
    x = sc_data["seconds"]
    y = sc_data["femto"]
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
    plt.ylabel(f"NPET clock to PPS difference [{unit}]")
    plt.xlabel("Measurement second [s]")
    plt.title(f"PPS Plot - {name}")
    plt.legend()
    plt.grid()
    plt.savefig(get_plot_path(f"pps_{name}"))
    plt.show(block=False)


def __plot(data: NPETData, name: str) -> None:
    """
    Plot the PPS data, using a different plotting method depending on the size of the data.
    :param data: Data to be plotted as NPETData object.
    :param name: Name of the file.
    """
    mean, mean_unit = data.sc_mean
    typer.secho(f"{name} mean delay = {mean:.5f} {mean_unit}", bold=True)
    std, std_unit = data.sc_std
    typer.echo(f"{name} STD = {std:.4f} {std_unit}")
    if len(data) > 600:
        __plot_long(data, name)
    else:
        __plot_short(data, name)


def main_pps() -> None:
    """Plot the PPS measurement. Used to evaluate the PPS stability."""
    try:
        pps_file_path: Path = user_file_select()
    except FileNotFoundError:
        return
    typer.echo(f"Importing data from {pps_file_path}")
    data: NPETData = NPETData.from_path(pps_file_path)
    processed: NPETData = data.process_incremental_overflow()
    typer.echo()
    __plot(processed, name=pps_file_path.stem)
    # Plot Allan time deviation
    if not processed.is_seconds_continuous():
        typer.secho("Data not continuous, skipping TDEV!", fg=typer.colors.RED)
        return
    typer.echo()
    typer.echo("Plotting TDEV...")
    drift_compensated, deg = drift_removal_prompt(processed)
    plot_time_deviation(drift_compensated.structured_arr, 1, name=pps_file_path.stem)
    if deg:
        typer.echo("Plotting PPS data after drift removal...")
        name: str = f"{pps_file_path.stem} without pol deg {deg} drift"
        __plot(drift_compensated, name=name)
