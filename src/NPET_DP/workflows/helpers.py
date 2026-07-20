import math

import numpy as np
import typer
from click import Choice
from numpy.typing import NDArray

from NPET_DP.framework.config import config
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.helpers import auto_scale_num, get_unit
from NPET_DP.processing.plotting import plot_histogram


def drift_removal_prompt(data: NPETData) -> tuple[NPETData, int]:
    """
    Prompt the user for optional drift removal from data.
    :param data: Data to remove drift from
    :return: Data with drift removed, if applicable by the user input.
    """
    pol_deg: int = typer.prompt(
        "Enter the polynomial degree for drift removal (0 - no drift compensation)",
        type=Choice([0, 1, 2]),
        default=0,
        show_choices=False,
    )
    if len(data) < pol_deg + 1:
        typer.secho(
            f"Required >={pol_deg + 1} points in data, skipping drift removal ...",
            fg=typer.colors.RED,
        )
        return data, 0
    if pol_deg == 0:
        return data, 0
    typer.echo(f"Removing drift with polynomial degree {pol_deg} ...")
    return data.compensate_drift(pol_deg), pol_deg


def auto_range(delays: NPETData, signal: NDArray[np.bool_]) -> None:
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


def select_data_within_range(data: NPETData) -> NPETData:
    """
    Select data within a specified range, uses the values stored in config.py.
    If there is no range stored in config yet, the user is prompted to enter one.
    :param data: The data to be filtered, as NPETData object.
    :return: Filtered data, as NPETData object.
    """
    mask = (config.min_delay <= data.femto) & (data.femto <= config.max_delay)
    return data.filter_range(mask)


def get_bin_count(data: NPETData, target_bin_size_fs: int) -> int:
    """
    Calculate the number of bins for a histogram based on the data and target bin size.
    :param data: Data to calculate the bin count for.
    :param target_bin_size_fs: Target bin size in femtoseconds.
    :return: Number of bins for the histogram.
    """
    # Difference between the max and min delay
    delay_spread: float = data.femto.max() - data.femto.min()
    if delay_spread > 50_000_000:  # fs
        # If there's too much data (>50 ns), then adjust the bin size to match 1000 bins
        bin_size: float = delay_spread / 1000
    else:
        # Otherwise, use the supplied bin size in femtoseconds
        bin_size: float = target_bin_size_fs
    typer.echo(f"Histogram bin size = {bin_size:.2f} fs")
    bin_count: int = math.floor(delay_spread / bin_size)
    typer.echo(f"Histogram bin count = {bin_count}")
    return bin_count


def histogram_plot_loop(data: NPETData, name: str) -> NPETData:
    """
    Loop to continuously filter and plot the histogram of the data.
    After each iteration, the user is prompted to adjust the x-axis range.
    :param data: Data to be filtered, as NPETData object.
    :param name: Name of the file.
    :return: Sigma filtered data, as NPETData object.
    """
    while True:
        selection: NPETData = select_data_within_range(data)
        # Apply the recursive sigma filter to the data
        sigma_data, sigma_i = selection.recursive_sigma_filter(config.sigma)
        typer.echo(f"\nRecursive {config.sigma} sigma filter results:")
        sc_mean, mean_unit = sigma_data.sc_mean
        sc_std, std_unit = sigma_data.sc_std
        typer.secho(f"Mean: {sc_mean:.4f} {mean_unit}", fg=typer.colors.CYAN)
        typer.secho(f"STD: {sc_std:.4f} {std_unit}", fg=typer.colors.CYAN)
        typer.echo(f"Accepted values in filtering = {len(sigma_data)}")
        typer.echo(f"Rejected values = {len(selection) - len(sigma_data)}")
        typer.echo(f"Number of iterations = {sigma_i}")
        ret_rate: float = len(sigma_data) / len(data)
        typer.secho(f"Return rate: {ret_rate:.2%}", fg=typer.colors.CYAN)
        # Plot the histogram of the filtered data
        typer.echo("\nPlotting histogram of the measured delays")
        bin_count = get_bin_count(selection, 10_000)
        plot_histogram(
            all_data=selection,
            signal_data=sigma_data if sigma_i != 1 else NPETData.empty(),
            name=name,
            bin_count=bin_count,
        )
        if not typer.confirm("Do you wish to adjust the x-axis range?"):
            break
        config.prompt_delay("min", validate=False)
        config.prompt_delay("max")
    return sigma_data