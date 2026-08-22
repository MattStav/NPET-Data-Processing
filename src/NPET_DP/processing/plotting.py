import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy.typing import NDArray

from NPET_DP.framework.config import config
from NPET_DP.framework.constants import FEMTO
from NPET_DP.framework.path_handler import get_plot_path
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.helpers import (
    auto_scale_data,
    auto_scale_num,
    get_unit,
    scale_data,
    scale_num,
)


def plot_time_deviation(data: NPETData, frequency: int, name: str) -> None:
    """Calculate and plot the time deviation of the data.

    :param data: Data to be plotted, as NPETData object.
    :param frequency: Frequency of the data
    :param name: Name of the file.
    """
    # The allantools package is poorly made, and its import is very timely, hence the lazy import
    from allantools import tdev

    assert frequency > 0, f"Frequency must be positive: {frequency}"
    assert name, "Name must not be empty"
    # Calculate TDEV
    taus, tdevs, errors, _ = tdev(data.femto / FEMTO, taus="octave", rate=frequency)
    # Scale the data to reasonable numbers
    sc_tdevs: NDArray[np.floating]
    sc_tdevs, sc_iter = auto_scale_data(tdevs)
    sc_errors: NDArray[np.floating] = scale_data(errors, sc_iter)
    # Calculate error bounds
    lower = np.maximum(sc_tdevs - sc_errors, np.finfo(float).tiny)
    upper = sc_tdevs + sc_errors
    _, ax = plt.subplots()
    ax.loglog(
        taus,
        sc_tdevs,
        "-",
        color="tab:blue",
        linewidth=1.8,
        markersize=4,
        label="TDEV",
    )
    ax.fill_between(
        taus,
        lower,
        upper,
        color="tab:blue",
        alpha=0.2,
        label="Uncertainty",
    )
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[*range(2, 10)]))
    ax.yaxis.set_minor_formatter(
        ticker.FuncFormatter(
            lambda x, p: (
                (f"{x:.1f}" if x % 1 else f"{int(x)}")
                if round(x / 10 ** np.floor(np.log10(x))) % 2 == 0
                else ""
            )
        )
    )
    for label in ax.yaxis.get_minorticklabels():
        label.set_color("gray")
        label.set_fontsize(8)
    ax.set_title(f"Time Deviation - {name}")
    ax.set_xlabel("Averaging time τ [s]")
    ax.set_ylabel(f"TDEV [{get_unit('s', sc_iter)}]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    plt.savefig(get_plot_path(f"tdev_{name}"))
    plt.show(block=False)


def plot_histogram(
    *,
    all_data: NPETData,
    signal_data: NPETData,
    name: str,
    bin_count: int,
) -> None:
    """Plot a histogram of the measured data.

    :param all_data: All measured data as a NPETData object.
    :param signal_data: Signal data as an NPETData object.
    :param name: Name of the plot.
    :param bin_count: Number of bins for the histogram.
    """
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
    counts, bins, _ = plt.hist(
        hist_data,
        bins=bins,
        color=hist_colors,
        alpha=0.7,
        label=hist_labels,
        stacked=True,
    )
    # Mark FWHM in the figure
    fwhm, peak_arg = all_data.calc_fwhm()
    # Scale to the same units as the (already scaled) plot axis
    sc_max_arg: float = scale_num(peak_arg, sc_iter)
    sc_fwhm: float = scale_num(fwhm, sc_iter)
    peak_bin: int = int(
        np.clip(
            np.searchsorted(bins, sc_max_arg, side="right") - 1,
            0,
            len(bins) - 2,
        )
    )
    half_max: float = float(np.asarray(counts)[-1][peak_bin] / 2)
    offset: float = sc_fwhm * 0.8  # tweak the fraction to taste
    x_left: float = sc_max_arg - sc_fwhm / 2 - offset / 2
    x_right: float = sc_max_arg + sc_fwhm / 2 + offset / 2
    plt.annotate(
        "",
        xy=(x_left, half_max),
        xytext=(x_left - offset, half_max),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
    )
    plt.annotate(
        "",
        xy=(x_right, half_max),
        xytext=(x_right + offset, half_max),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
    )
    auto_sc_fwhm, auto_fwhm_iter = auto_scale_num(fwhm)
    plt.text(
        x_right + offset * 1.15,
        half_max,
        f"FWHM = {auto_sc_fwhm:.2f} {get_unit('fs', auto_fwhm_iter)}",
        color="black",
        ha="left",
        va="center",
        fontsize=11,
    )
    # Mark the Gaussian curve in the figure
    if len(signal_data) != 0:
        signal_mean: float = float(signal_data.femto.mean())
        sc_mean, sc_mean_iter = auto_scale_num(signal_mean)
        signal_std: float = float(signal_data.femto.std())
        sc_std, sc_std_iter = auto_scale_num(signal_std)
        x = np.linspace(sc_bgr.min(), sc_bgr.max(), bin_count * 10)
        # Correction for the STD being in different units
        std_correct = scale_num(sc_std, sc_mean_iter - sc_std_iter)
        gaussian: NDArray[np.floating] = np.exp(
            -((x - sc_mean) ** 2) / (2 * std_correct**2)
        )
        max_gaussian = np.max(gaussian)
        max_counts: np.floating = np.max(np.asarray(counts)[-1])
        gaussian *= max_counts / max_gaussian
        plt.plot(
            x,
            gaussian,
            "k",
            linewidth=2,
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
