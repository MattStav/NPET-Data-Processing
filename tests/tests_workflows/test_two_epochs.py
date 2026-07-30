from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from NPET_DP.framework.config import config
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import (
    auto_range,
    get_bin_count,
    select_data_within_range,
)
from NPET_DP.workflows.two_epochs import __plot_all_scatter, main_two_epochs


def test_returns_when_no_files() -> None:
    """Test return when no file.

    Test that workflow returns when there is no file to process.
    """
    with (
        patch(
            "NPET_DP.workflows.two_epochs.user_file_select",
            side_effect=FileNotFoundError,
        ),
        patch("NPET_DP.workflows.two_epochs.NPETData.from_path") as mock_from_path,
    ):
        main_two_epochs()
    mock_from_path.assert_not_called()


def test_auto_range() -> None:
    """Test autoranging.

    Test that auto_range returns correct range, defined by the signal within the data.
    """
    delays = NPETData(
        seconds=np.zeros(5, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200, 300, 400, 500]),
    )
    mask = np.array([False, True, True, True, False])
    auto_range(delays, mask)
    assert config.min_delay == -340.0
    assert config.max_delay == 1260.0


def test_select_data_within_range() -> None:
    """Test data selection with a simple case.

    The function should only select the data within the range defined by config.
    """
    config.min_delay = 200.0
    config.max_delay = 400.0
    data = NPETData(
        seconds=np.array([1, 2, 3, 4, 5]),
        femto=np.array([100, 200, 300, 400, 500]),
    )
    result = select_data_within_range(data)
    assert np.array_equal(result.femto, np.array([200, 300, 400]))


@pytest.mark.parametrize(
    ("femto_max", "target_bin_size_fs", "expected_bin_count"),
    [
        pytest.param(25_000, 10_000, 2, id="small_spread_uses_target_bin_size"),
        pytest.param(60_000_000, 10_000, 1000, id="large_spread_scales_bin_size"),
        pytest.param(
            50_000_000, 10_000, 5000, id="boundary_spread_uses_target_bin_size"
        ),
        pytest.param(5_000, 1_000, 5, id="custom_target_bin_size"),
    ],
)
def test_get_bin_count(
    femto_max: int,
    target_bin_size_fs: int,
    expected_bin_count: int,
) -> None:
    """Test bin count calculation across delay spreads and target bin sizes.

    Spreads at or below 50,000,000 fs use the target bin size directly;
    spreads above that scale the bin size so the count is always 1000 bins.
    """
    data = NPETData(
        seconds=np.zeros(2, dtype=np.int_).astype(np.int_),
        femto=np.array([0, femto_max]),
    )
    assert (
        get_bin_count(data, target_bin_size_fs=target_bin_size_fs) == expected_bin_count
    )


@patch("NPET_DP.workflows.two_epochs.show")
def test_plot_all_delays_interactive(
    mock_show: MagicMock,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that plotting all data in interactive scatter works."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    data = NPETData(
        seconds=np.zeros(2, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200]),
    )
    masks = (np.array([True, False]),)
    __plot_all_scatter(data, masks)
    mock_show.assert_called_once()
