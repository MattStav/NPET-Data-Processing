import numpy as np
from allantools import tdev
from matplotlib import pyplot as plt
from matplotlib import ticker
from numpy.typing import NDArray

from NPET_DP.processing.helpers import (
    auto_scale_data,
    get_unit,
    scale_data,
    validate_inputs,
)
from NPET_DP.framework.constants import FEMTO
from NPET_DP.framework.path_handler import get_plot_path


@validate_inputs
def plot_time_deviation(data: NDArray, frequency: int, name: str) -> None:
    """
    Calculate and plot the time deviation of the data.
    :param data: Data to be plotted, in the FW standard format.
    :param frequency: Frequency of the data
    :param name: Name of the file
    """
    assert frequency > 0, f"Frequency must be positive: {frequency}"
    assert name, "Name must not be empty"
    # Calculate TDEV
    femto_in_seconds: NDArray = data["femto"] / FEMTO
    taus, tdevs, errors, _ = tdev(femto_in_seconds, taus="octave", rate=frequency)
    # Scale the data to reasonable numbers
    sc_tdevs: NDArray[np.floating]
    sc_tdevs, scaled_num = auto_scale_data(tdevs)
    sc_errors: NDArray[np.floating] = scale_data(errors, scaled_num)
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
                if int(round(x / 10 ** np.floor(np.log10(x)))) % 2 == 0
                else ""
            )
        )
    )
    for label in ax.yaxis.get_minorticklabels():
        label.set_color("gray")
        label.set_fontsize(8)
    ax.set_title(f"Time Deviation - {name}")
    ax.set_xlabel("Averaging time τ [s]")
    ax.set_ylabel(f"TDEV [{get_unit('s', scaled_num)}]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    plt.savefig(get_plot_path(f"tdev_{name}"))
    plt.show(block=False)
