import math
import webbrowser
from pathlib import Path
from typing import NamedTuple

import numpy as np
import typer
from bokeh.io import output_file
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from NPET_DP.processing.helpers import (
    auto_scale_data,
    auto_scale_num,
    get_unit,
    scale_data,
    scale_num,
)
from NPET_DP.processing.plotting import plot_time_deviation
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import drift_removal_prompt
from NPET_DP.framework.config import config
from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path

_Data = NamedTuple("_Data", [("data_start", NPETData), ("data_stop", NPETData)])


def __match_data(*, data_start: NPETData, data_stop: NPETData) -> _Data:
    """
    Discard data that was taken before the second (either start or stop) source begun measurement,
    until the data begins at the same second.
    :param data_start: Data from the start source, as NPETData object.
    :param data_stop: Data from the stop source, as NPETData object.
    :return: Both datasets, sliced to be within the same time frame.
    :raises IndexError: If no matching data is found.
    """
    typer.echo("\nDiscarding data from before the second source begun measurement")
    # Discard data until both datasets start at the same second
    data_stop_filtered, discarded = data_stop.discard_rows_until_ref_match(data_start)
    if discarded > 0:
        typer.echo(f"Discarded {discarded} epochs from STOP data")
        return _Data(data_start=data_start, data_stop=data_stop_filtered)
    data_start_filtered, discarded = data_start.discard_rows_until_ref_match(data_stop)
    if discarded > 0:
        typer.echo(f"Discarded {discarded} epochs from START data")
        return _Data(data_start=data_start_filtered, data_stop=data_stop)
    typer.echo("No data was discarded, both datasets start at the same time")
    return _Data(data_start=data_start, data_stop=data_stop)


def __plot_all_scatter(data: NPETData, signal: tuple[NDArray[np.bool_], ...]) -> None:
    """
    Plot all delay data on y-axis in the order of events.
    Uses interactive high-performance bokeh graphs.
    :param data: Data to be plotted as NPETData object.
    :param signal: List of boolean masks indicating detected signals
    """
    typer.echo("\nPlotting all the calculated delay data in interactive mode")
    output_file(get_plot_path("bokeh_plot", suffix="html"), mode="inline")
    sc_femto, unit = data.sc_femto
    fig = figure(
        title="Measured delay between START and STOP data",
        x_axis_label="Event Number [n]",
        y_axis_label=f"Time Difference [{unit}]",
        sizing_mode="stretch_both",  # Makes the plot size responsive to the window size
        background_fill_color="white",  # Set a white background
    )
    # Enable and style the grid
    fig.grid.grid_line_color = "gray"  # Grid line color
    fig.grid.grid_line_alpha = 0.3  # Grid line transparency
    fig.grid.grid_line_dash = [6, 4]  # Dashed grid lines
    fig.grid.visible = True  # Make sure the grid is visible
    # Enable minor grid lines
    fig.xgrid.minor_grid_line_color = "gray"
    fig.xgrid.minor_grid_line_alpha = 0.1
    fig.ygrid.minor_grid_line_color = "gray"
    fig.ygrid.minor_grid_line_alpha = 0.1
    # Add a hover tool that shows the event number and delay
    hover = HoverTool(
        tooltips=[
            ("Event Number", "@x"),
            ("Delay (ns)", "@y{0.000}"),  # Format with 3 decimal places
        ]
    )
    fig.add_tools(hover)
    # Create a combined mask for all detected signals
    combined_mask = np.zeros(len(data), dtype=bool)
    for mask in signal:
        combined_mask |= mask
    # Plot non-signal points first (background data)
    non_signal_indices: NDArray = np.where(~combined_mask)[0]
    if len(non_signal_indices) > 0:
        fig.scatter(
            non_signal_indices,
            sc_femto[non_signal_indices],
            size=2,
            color="navy",
            alpha=0.5,
            legend_label="Background data",
        )
    # Plot each detected signal with a different color
    colors = ["red", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    for i, mask in enumerate(signal):
        signal_indices: NDArray = np.where(mask)[0]
        if len(signal_indices) > 0:
            color = colors[i % len(colors)]
            fig.scatter(
                signal_indices,
                sc_femto[signal_indices],
                size=3,
                color=color,
                alpha=0.8,
                legend_label=f"Detected signal {i + 1}",
            )
    # Font sizes
    fig.title.text_font_size = "24pt"  # ty:ignore[invalid-assignment]
    fig.xaxis.axis_label_text_font_size = "20pt"
    fig.yaxis.axis_label_text_font_size = "20pt"
    fig.xaxis.major_label_text_font_size = "20pt"
    fig.yaxis.major_label_text_font_size = "20pt"
    fig.legend.label_text_font_size = "20pt"
    fig.legend.title_text_font_size = "22pt"
    # Show the plot in the system's default browser
    try:
        show(fig)
    except webbrowser.Error:
        typer.secho("Couldn't open the graph in interactive mode.", fg=typer.colors.RED)


def __auto_range(delays: NPETData, signal: NDArray[np.bool_]) -> None:
    """
    Automatically set the x-axis range of the histogram to focus on the detected signal.
    :param delays: Data to be filtered as NPETData object.
    :param signal: Boolean mask indicating the detected signal.
    """
    typer.echo("\nAuto-ranging the histogram plot to focus on the detected signal")
    signal_range = delays.define_signal_range(signal)
    config.min_delay = signal_range[0]
    config.max_delay = signal_range[1]
    r_min, n_min = auto_scale_num(config.min_delay)
    r_max, n_max = auto_scale_num(config.max_delay)
    u_min = get_unit("fs", n_min)
    u_max = get_unit("fs", n_max)
    typer.echo(f"Range set to min: {r_min:.4f} {u_min}, max: {r_max:.4f} {u_max}")


def __select_data_within_range(data: NPETData) -> NPETData:
    """
    Select data within a specified range, uses the values stored in config.py.
    :param data: The data to be filtered, in the FW standard format.
    :return: Filtered data, in the FW standard format.
    """
    mask = (config.min_delay <= data.femto) & (data.femto <= config.max_delay)
    return data.filter_range(mask)


def __plot_histogram(
    *,
    all_data: NPETData,
    signal_data: NPETData,
    name: str,
    bin_size_fs: int = 10_000,  # 0.01 ns
) -> None:
    """
    Plot a histogram of the measured data.
    :param all_data: All measured data as a NPETData object.
    :param signal_data: Signal data as an NPETData object.
    :param name: Name of the plot.
    :param bin_size_fs: Bin size in fs.
    """
    typer.echo("\nPlotting histogram of the measured delays")
    # Difference between the max and min delay
    delay_spread: float = all_data.femto.max() - all_data.femto.min()
    if delay_spread > 50_000_000:
        # If there's too much data (>50 ns), then adjust the bin size to match 1000 bins
        bin_size: float = delay_spread / 1000
    else:
        # Otherwise, use the supplied bin size in femtoseconds
        bin_size: float = bin_size_fs
    typer.echo(f"Histogram bin size = {bin_size:.2f} fs")
    bin_count: int = math.floor(delay_spread / bin_size)
    typer.echo(f"Histogram bin count = {bin_count}")
    # Scale the data for plotting
    bgr_data = all_data.femto_not_in(signal_data)
    sc_bgr, sc_iter = auto_scale_data(bgr_data.femto)
    # Create bins based on bin size
    bins = np.linspace(sc_bgr.min(), sc_bgr.max(), bin_count)
    plt.clf()
    hist_data = []
    hist_labels = []
    hist_colors = []
    # If there is data denoting the signal, plot it
    if len(signal_data) != 0:
        sc_signal = scale_data(signal_data.femto, sc_iter)
        hist_data.append(sc_signal)
        hist_labels.append(f"Recursive Gauss filtered ({config.sigma}σ)")
        hist_colors.append("red")
    hist_data.append(sc_bgr)
    hist_labels.append("Other measured data")
    hist_colors.append("blue")
    counts, _, _ = plt.hist(
        hist_data,
        bins=bins,
        color=hist_colors,
        alpha=0.7,
        label=hist_labels,
        stacked=True,
    )
    if len(signal_data) != 0:
        # Add the Gaussian curve
        sc_mean, sc_mean_iter = auto_scale_num(signal_data.femto.mean())
        sc_std, sc_std_iter = auto_scale_num(signal_data.femto.std())
        x = np.linspace(sc_bgr.min(), sc_bgr.max(), bin_count * 10)
        # Correction for the STD being in different units
        std_correct = scale_num(sc_std, sc_mean_iter - sc_std_iter)
        gaussian = np.exp(-((x - sc_mean) ** 2) / (2 * std_correct**2))
        gaussian *= np.max(counts) / np.max(gaussian)
        plt.plot(
            x,
            gaussian,
            "k",
            linewidth=1,
            label=f"Gaussian \n"
            f"μ={sc_mean:.3f} {get_unit('fs', sc_mean_iter)}\n"
            f"σ={sc_std:.3f} {get_unit('fs', sc_std_iter)}",
            alpha=0.5,
        )
    plt.title(f"Histogram - {name}")
    plt.xlabel(f"Delay [{get_unit('fs', sc_iter)}]")
    plt.ylabel("Counts [n]")
    plt.grid(True)
    plt.legend()
    plt.savefig(get_plot_path(f"histogram_{name}"))
    plt.show(block=False)


def main_two_epochs() -> None:
    """Process data from two epochs, used to compare measurements from two different epochs."""
    col = typer.colors
    try:
        start_file: Path = user_file_select("START signal")
        stop_file: Path = user_file_select("STOP signal", [start_file])
    except FileNotFoundError:
        return
    typer.echo(f"Importing data from {start_file}")
    data_start: NPETData = NPETData.from_path(start_file)
    typer.echo(f"Importing data from {stop_file}")
    data_stop: NPETData = NPETData.from_path(stop_file)
    try:
        matching_data: _Data = __match_data(data_start=data_start, data_stop=data_stop)
    except IndexError:
        typer.secho("No matching signals found!", fg=col.RED)
        return
    typer.echo("\nCalculating the delay between the START and STOP measured data")
    frequency: int = config.frequency
    delays: NPETData = data_start.calc_delay_start(
        stop=matching_data.data_stop,
        frequency=frequency,
    )
    typer.secho(f"Number of accepted values: {len(delays)}", fg=col.GREEN)
    typer.echo("\nAttempting to autodetect the return signal")
    autodetection: tuple[NDArray[np.bool_], ...] = delays.detect_signal()
    if len(autodetection) == 0:
        typer.secho("Failed to autodetect any return signal!", fg=col.RED)
        if not typer.confirm("No signal detected! Do you wish to proceed anyway?"):
            return
    elif len(autodetection) == 1:
        typer.secho("Autodetect found a single return signal!", fg=col.GREEN)
        __auto_range(delays, autodetection[0])
    elif len(autodetection) > 1:
        typer.secho(f"Autodetect found {len(autodetection)} signals!", fg=col.YELLOW)
    __plot_all_scatter(delays, autodetection)
    while True:
        sel_del: NPETData = __select_data_within_range(delays)
        # Apply the recursive sigma filter to the data
        sigma_del, sigma_i = sel_del.recursive_sigma_filter(config.sigma)
        typer.echo(f"\nRecursive {config.sigma} sigma filter results:")
        sc_mean, mean_unit = sigma_del.sc_mean
        sc_std, std_unit = sigma_del.sc_std
        typer.secho(f"Mean: {sc_mean:.4f} {mean_unit}", fg=typer.colors.CYAN)
        typer.secho(f"STD: {sc_std:.4f} {std_unit}", fg=typer.colors.CYAN)
        typer.echo(f"Accepted values in filtering = {len(sigma_del)}")
        typer.echo(f"Rejected values = {len(sel_del) - len(sigma_del)}")
        typer.echo(f"Number of iterations = {sigma_i}")
        ret_rate: float = len(sigma_del) / len(delays)
        typer.secho(f"Return rate: {ret_rate:.2%}", fg=typer.colors.CYAN)
        # Plot the histogram of the filtered data
        __plot_histogram(
            all_data=sel_del,
            signal_data=sigma_del if sigma_i != 1 else NPETData.empty(),
            name=stop_file.stem,
        )
        if not typer.confirm("Do you wish to adjust the x-axis range?"):
            break
        config.prompt_delay("min", validate=False)
        config.prompt_delay("max")
    no_drift, deg = drift_removal_prompt(sigma_del)
    name: str = stop_file.stem
    if deg > 0:
        name += f" without pol deg {deg} drift"
    plot_time_deviation(no_drift.structured_arr, frequency, name)
    plt.close()
