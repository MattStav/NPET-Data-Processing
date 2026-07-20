import numpy as np
from unittest.mock import patch, MagicMock

from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.two_epochs import (
    main_two_epochs,
    __auto_range,
    __select_data_within_range,
    __plot_histogram,
    __plot_all_scatter,
)
from NPET_DP.framework.config import config


def test_returns_when_no_files():
    """Test that the function returns when there is no file to process"""
    with (
        patch(
            "NPET_DP.workflows.two_epochs.user_file_select",
            side_effect=FileNotFoundError,
        ),
        patch("NPET_DP.workflows.two_epochs.NPETData.from_path") as mock_from_path,
    ):
        main_two_epochs()
    mock_from_path.assert_not_called()


def test_auto_range():
    """Test the __auto_range function with a simple case"""
    delays = NPETData(
        seconds=np.zeros(5, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200, 300, 400, 500]),
    )
    mask = np.array([False, True, True, True, False])
    __auto_range(delays, mask)
    assert config.min_delay == -340.0
    assert config.max_delay == 1260.0


def test_select_data_within_range():
    """Test the __select_data_within_range function with a simple case"""
    config.min_delay = 200.0
    config.max_delay = 400.0
    data = NPETData(
        seconds=np.array([1, 2, 3, 4, 5]),
        femto=np.array([100, 200, 300, 400, 500]),
    )
    result = __select_data_within_range(data)
    assert np.array_equal(result.femto, np.array([200, 300, 400]))


@patch("NPET_DP.workflows.two_epochs.plt")
@patch("NPET_DP.workflows.two_epochs.scale_num")
@patch("NPET_DP.workflows.two_epochs.auto_scale_num")
@patch("NPET_DP.workflows.two_epochs.scale_data")
@patch("NPET_DP.workflows.two_epochs.auto_scale_data")
def test_plot_histogram(
    mock_auto_scale_data,
    mock_scale_data,
    mock_auto_scale_num,
    mock_scale_num,
    mock_plt,
    monkeypatch,
    tmp_path,
):
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
        (250.0, 0),  # Mean
        (50.0, 0),  # Std
    ]
    mock_scale_num.return_value = 50.0  # Std correction
    # Just check if it runs without error and calls plt.show
    mock_plt.hist.return_value = (np.array([1, 1]), np.array([1, 2, 3]), MagicMock())
    # Mock np.exp and np.linspace to return non-empty arrays to avoid division by zero or empty max
    with (
        patch(
            "NPET_DP.workflows.two_epochs.np.exp",
            return_value=np.array([0.5, 1.0]),
        ),
        patch(
            "NPET_DP.workflows.two_epochs.np.linspace",
            return_value=np.array([1, 2]),
        ),
    ):
        __plot_histogram(all_data=all_data, signal_data=filtered, name="test")
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_called_once()


@patch("NPET_DP.workflows.two_epochs.plt")
@patch("NPET_DP.workflows.two_epochs.scale_data")
@patch("NPET_DP.workflows.two_epochs.scale_num")
def test_plot_histogram_empty_filtered(
    mock_scale_num,
    mock_scale_data,
    mock_plt,
    monkeypatch,
    tmp_path,
):
    """Test that __plot_histogram handles empty filtered data correctly"""
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
    # Just check if it runs without error and calls plt.show
    mock_plt.hist.return_value = (
        np.array([1, 1, 1]),
        np.array([100, 200, 300]),
        MagicMock(),
    )
    __plot_histogram(all_data=all_data, signal_data=filtered, name="test_empty")
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_called_once()
    # Ensure it didn't try to calculate mean/std of empty filtered array
    mock_scale_num.assert_not_called()


@patch("NPET_DP.workflows.two_epochs.show")
def test_plot_all_delays_interactive(
    mock_show,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    data = NPETData(
        seconds=np.zeros(2, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200]),
    )
    masks = (np.array([True, False]),)
    __plot_all_scatter(data, masks)
    mock_show.assert_called_once()
