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

from NPET_DP.framework.config import config
from NPET_DP.framework.file_selection import user_file_select
from NPET_DP.framework.path_handler import get_plot_path
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.helpers import auto_scale_data, get_unit
from NPET_DP.processing.plotting import plot_time_deviation
from NPET_DP.workflows.helpers import (
    auto_range,
    drift_removal_prompt,
    histogram_plot_loop,
)

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
    sc_femto, sc_iter = auto_scale_data(data.femto, 2)
    unit = get_unit("fs", sc_iter)
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
        auto_range(delays, autodetection[0])
    elif len(autodetection) > 1:
        typer.secho(f"Autodetect found {len(autodetection)} signals!", fg=col.YELLOW)
    __plot_all_scatter(delays, autodetection)
    sigma_filtered: NPETData = histogram_plot_loop(delays, stop_file.stem)
    no_drift, deg = drift_removal_prompt(sigma_filtered)
    name: str = stop_file.stem
    if deg > 0:
        name += f" without pol deg {deg} drift"
    plot_time_deviation(no_drift, frequency, name)
    plt.close()
