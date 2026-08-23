import numpy as np
import typer
from click import Choice
from numpy.typing import NDArray

from NPET_DP.framework.config import config
from NPET_DP.processing.calculations import get_bin_count
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.helpers import get_signal_xrange
from NPET_DP.processing.plotting import plot_histogram


def drift_removal_prompt(data: NPETData) -> tuple[NPETData, int]:
    """Prompt the user for optional drift removal from data.

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


def rough_auto_range(delays: NPETData, signal: NDArray[np.bool_]) -> None:
    """Auto range the x-axis based on the data.

    This ranging is pretty quick, but not very accurate.
    Automatically set the x-axis range of the histogram to focus on the detected signal.
    :param delays: Data to be filtered as NPETData object.
    :param signal: Boolean mask indicating the detected signal.
    """
    typer.echo("Auto-ranging to focus roughly on the detected signal")
    signal_delays = delays.filter_range(signal)
    sig_width, sig_place = signal_delays.calc_fwhm()
    signal_range = get_signal_xrange(sig_place, sig_width * 3)
    config.assign_delays(signal_range[0], signal_range[1])


def precise_auto_range(sig_place_fs: int, sig_width_fs: int) -> None:
    """Auto range the x-axis based on the precise location and width of the signal.

    :param sig_place_fs: The place of the signal in femtoseconds.
    :param sig_width_fs: The width of the signal in femtoseconds.
    """
    typer.echo("Auto-ranging to focus precisely on the calculated signal")
    signal_range = get_signal_xrange(sig_place_fs, sig_width_fs * 4)
    config.assign_delays(signal_range[0], signal_range[1])


def select_data_within_range(data: NPETData) -> NPETData:
    """Select data within a specified range.

    The selection uses the values stored in config.py.
    If there is no range stored in config yet, the user is prompted to enter one.
    :param data: The data to be filtered, as NPETData object.
    :return: Filtered data, as NPETData object.
    """
    mask = (config.min_delay <= data.femto) & (data.femto <= config.max_delay)
    return data.filter_range(mask)


def histogram_plot_loop(data: NPETData, name: str) -> NPETData:
    """Loop to continuously filter and plot the histogram of the data.

    Infinite loop that allows the user to filter the data using a recursive sigma filter and plot the histogram of the filtered data.
    After each iteration, the user is prompted to adjust the x-axis range.
    :param data: Data to be filtered, as NPETData object.
    :param name: Name of the file.
    :return: Sigma filtered data, as NPETData object.
    """
    auto_precise: bool = True
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
        if auto_precise:
            precise_auto_range(round(sigma_data.mean), round(sigma_data.std))
        selection = select_data_within_range(data)
        data_spread: float = selection.femto.max() - selection.femto.min()
        bin_count = get_bin_count(data_spread, 10_000)
        typer.echo(f"Histogram bin count = {bin_count}")
        plot_histogram(
            all_data=selection,
            signal_data=sigma_data if sigma_i != 1 else NPETData.empty(),
            name=name,
            bin_count=bin_count,
        )
        redraw = typer.prompt(
            "Rescale x-axis?",
            type=Choice(["manual", "auto", "no"]),
            default="no",
            show_choices=True,
        )
        if redraw == "no":
            break
        auto_precise = False
        if redraw == "manual":
            config.prompt_delay("min", validate=False)
            config.prompt_delay("max")
            continue
        # elif "auto"
        autodetection = selection.detect_signal()
        typer.echo(f"\nAutodetection found {len(autodetection)} signals")
        if len(autodetection) != 1:
            typer.secho("Failed to autodetect a single signal!", fg=typer.colors.RED)
            continue
        auto_precise = True
        rough_auto_range(selection, autodetection[0])

    return sigma_data
