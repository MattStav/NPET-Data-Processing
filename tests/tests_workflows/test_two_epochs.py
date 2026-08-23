from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from NPET_DP.framework.config import config
from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import (
    rough_auto_range,
    select_data_within_range,
)
from NPET_DP.workflows.two_epochs import (
    __match_data,
    __plot_all_scatter,
    main_two_epochs,
)


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


def test_rough_auto_range() -> None:
    """Test autoranging.

    Test that auto_range returns the correct range, defined by the signal within the data.
    """
    delays = NPETData(
        seconds=np.zeros(7, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200, 300, 400, 500, 400, 300]),
    )
    mask = np.array([False, True, True, True, True, True, False])
    rough_auto_range(delays, mask)
    assert config.min_delay == -1000
    assert config.max_delay == 4400


def test_select_data_within_range() -> None:
    """Test data selection with a simple case.

    The function should only select the data within the range defined by config.
    """
    config.min_delay = 200
    config.max_delay = 400
    data = NPETData(
        seconds=np.array([1, 2, 3, 4, 5]),
        femto=np.array([100, 200, 300, 400, 500]),
    )
    result = select_data_within_range(data)
    assert np.array_equal(result.femto, np.array([200, 300, 400]))


def test_match_data_discards_from_stop_when_stop_started_earlier() -> None:
    """Test match data discarding early STOP rows.

    When the STOP data begins before the START data, the leading STOP rows
    should be discarded and the START data left untouched.
    """
    data_start = NPETData(seconds=np.array([10, 11, 12]), femto=np.array([0, 0, 0]))
    data_stop = NPETData(
        seconds=np.array([8, 9, 10, 11, 12]),
        femto=np.array([1, 2, 3, 4, 5]),
    )
    result = __match_data(data_start=data_start, data_stop=data_stop)
    assert result.data_start is data_start
    assert np.array_equal(result.data_stop.seconds, np.array([10, 11, 12]))
    assert np.array_equal(result.data_stop.femto, np.array([3, 4, 5]))


def test_match_data_discards_from_start_when_start_started_earlier() -> None:
    """Test match data discarding early START rows.

    When the START data begins before the STOP data, the leading START rows
    should be discarded and the STOP data left untouched.
    """
    data_start = NPETData(
        seconds=np.array([5, 6, 10, 11, 12]),
        femto=np.array([1, 2, 3, 4, 5]),
    )
    data_stop = NPETData(seconds=np.array([10, 11, 12]), femto=np.array([0, 0, 0]))
    result = __match_data(data_start=data_start, data_stop=data_stop)
    assert result.data_stop is data_stop
    assert np.array_equal(result.data_start.seconds, np.array([10, 11, 12]))
    assert np.array_equal(result.data_start.femto, np.array([3, 4, 5]))


def test_match_data_keeps_both_when_already_aligned() -> None:
    """Test match data with already-aligned datasets.

    When both datasets already start at the same second, neither dataset
    should be modified.
    """
    data_start = NPETData(seconds=np.array([10, 11, 12]), femto=np.array([1, 2, 3]))
    data_stop = NPETData(
        seconds=np.array([10, 11, 12, 13]),
        femto=np.array([4, 5, 6, 7]),
    )
    result = __match_data(data_start=data_start, data_stop=data_stop)
    assert result.data_start is data_start
    assert result.data_stop is data_stop


def test_match_data_raises_when_no_overlap_found() -> None:
    """Test match data with no overlap.

    When the STOP data never reaches the START data's first second,
    matching should raise an IndexError.
    """
    data_start = NPETData(seconds=np.array([100]), femto=np.array([1]))
    data_stop = NPETData(seconds=np.array([1, 2, 3]), femto=np.array([1, 2, 3]))
    with pytest.raises(IndexError):
        __match_data(data_start=data_start, data_stop=data_stop)


@patch("NPET_DP.workflows.two_epochs.show")
def test_plot_all_delays_interactive(
    mock_show: MagicMock,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that plotting all data in an interactive scatter works."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    data = NPETData(
        seconds=np.zeros(2, dtype=np.int_).astype(np.int_),
        femto=np.array([100, 200]),
    )
    masks = (np.array([True, False]),)
    __plot_all_scatter(data, masks)
    mock_show.assert_called_once()
