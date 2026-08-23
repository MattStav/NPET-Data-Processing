from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from NPET_DP.framework.config import config
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.plotting import plot_histogram
from NPET_DP.workflows.helpers import get_bin_count


@patch("NPET_DP.processing.plotting.plt")
@patch("NPET_DP.processing.plotting.scale_num")
@patch("NPET_DP.processing.plotting.auto_scale_num")
@patch("NPET_DP.processing.plotting.scale_data")
@patch("NPET_DP.processing.plotting.auto_scale_data")
def test_plot_histogram(
    mock_auto_scale_data: MagicMock,
    mock_scale_data: MagicMock,
    mock_auto_scale_num: MagicMock,
    mock_scale_num: MagicMock,
    mock_plt: MagicMock,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test the histogram plotting.

    Check that when plotting data in the histogram, it does not fail, a histogram is shown and saved.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    all_data_arr = np.array([100, 200, 300])
    filtered_arr = np.array([200, 300])
    all_data = NPETData(
        seconds=np.zeros(3, dtype=np.int_).astype(np.int_),
        femto=all_data_arr,
    )
    filtered = NPETData(
        seconds=np.zeros(2, dtype=np.int_).astype(np.int_),
        femto=filtered_arr,
    )
    config.sigma = 2.2
    # simplified non_filtered
    mock_auto_scale_data.return_value = (all_data_arr[all_data_arr != 200], 0)
    mock_scale_data.return_value = filtered_arr
    mock_auto_scale_num.side_effect = [
        (100.0, 0),  # FWHM annotation
        (250.0, 0),  # Mean
        (50.0, 0),  # Std
    ]
    mock_scale_num.return_value = 50.0  # Std correction
    # Just check if it runs without error and calls plt.show
    # stacked=True with 2 datasets (signal + background) makes plt.hist return
    # counts as one row per dataset, so plotting.py's counts[-1] indexing works.
    mock_plt.hist.return_value = (
        np.array([[1, 1], [1, 1]]),
        np.array([1, 2, 3]),
        MagicMock(),
    )
    # Mock np.exp and np.linspace to return non-empty arrays to avoid division by zero or empty max
    spread = all_data.femto.max() - all_data.femto.min()
    bin_count = get_bin_count(spread)
    with (
        patch(
            "NPET_DP.processing.plotting.np.exp",
            return_value=np.array([0.5, 1.0]),
        ),
        patch(
            "NPET_DP.processing.plotting.np.linspace",
            return_value=np.array([1, 2]),
        ),
    ):
        plot_histogram(
            all_data=all_data,
            signal_data=filtered,
            name="test",
            bin_count=bin_count,
        )
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_called_once()


@patch("NPET_DP.processing.plotting.plt")
@patch("NPET_DP.processing.plotting.scale_data")
@patch("NPET_DP.processing.plotting.scale_num")
def test_plot_histogram_empty_filtered(
    mock_scale_num: MagicMock,
    mock_scale_data: MagicMock,
    mock_plt: MagicMock,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test plot_histogram with empty filtered data.

    Test that plot_histogram handles empty filtered data correctly and does not fail.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    all_data_arr = np.array([100, 200, 300])
    all_data = NPETData(
        seconds=np.zeros(3, dtype=np.int_).astype(np.int_),
        femto=all_data_arr,
    )
    filtered = NPETData.empty()
    config.sigma = 2.2
    # scale_data will be called once for non_filtered_data (which is all_data here)
    mock_scale_data.return_value = (all_data_arr, 0)
    # Used unconditionally for the FWHM annotation, even with no signal data
    mock_scale_num.return_value = 150.0
    # Just check if it runs without error and calls plt.show
    # stacked=True with 1 dataset (background only, no signal) still makes plt.hist
    # return counts as a single row, so plotting.py's counts[-1] indexing works.
    mock_plt.hist.return_value = (
        np.array([[1, 1, 1]]),
        np.array([100, 200, 300]),
        MagicMock(),
    )
    spread = all_data.femto.max() - all_data.femto.min()
    bin_count = get_bin_count(spread)
    plot_histogram(
        all_data=all_data,
        signal_data=filtered,
        name="test_empty",
        bin_count=bin_count,
    )
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_called_once()
    # Ensure it didn't try to plot the Gaussian curve for the empty filtered array
    mock_plt.plot.assert_not_called()
