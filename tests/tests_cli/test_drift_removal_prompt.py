from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import CaptureFixture

from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import drift_removal_prompt


def _make_data(num_points: int) -> NPETData:
    """Build minimal NPETData with the given number of rows."""
    return NPETData(
        seconds=np.zeros(num_points, dtype=np.int_).astype(np.int_),
        femto=np.arange(num_points, dtype=np.int_).astype(np.int_),
    )


@patch("NPET_DP.workflows.helpers.NPETData.compensate_drift")
@patch("typer.prompt")
def test_drift_removal_prompt_no_compensation_selected(
    mock_prompt: MagicMock,
    mock_compensate_drift: MagicMock,
) -> None:
    """Test that selecting polynomial degree 0 leaves the data untouched."""
    mock_prompt.return_value = 0
    data = _make_data(5)
    result_data, result_deg = drift_removal_prompt(data)
    assert result_data is data
    assert result_deg == 0
    mock_compensate_drift.assert_not_called()


@pytest.mark.parametrize("pol_deg", [1, 2])
@patch("NPET_DP.workflows.helpers.NPETData.compensate_drift")
@patch("typer.prompt")
def test_drift_removal_prompt_skips_when_not_enough_points(
    mock_prompt: MagicMock,
    mock_compensate_drift: MagicMock,
    pol_deg: int,
    capsys: CaptureFixture,
) -> None:
    """Test that too few points for the chosen degree skips drift removal.

    Skipping must happen even though a nonzero polynomial degree was chosen,
    since there aren't enough points in the data to fit it.
    """
    mock_prompt.return_value = pol_deg
    data = _make_data(pol_deg)  # one point short of the pol_deg + 1 required
    result_data, result_deg = drift_removal_prompt(data)
    assert result_data is data
    assert result_deg == 0
    mock_compensate_drift.assert_not_called()
    assert "skipping drift removal" in capsys.readouterr().out


@pytest.mark.parametrize("pol_deg", [1, 2])
@patch("NPET_DP.workflows.helpers.NPETData.compensate_drift")
@patch("typer.prompt")
def test_drift_removal_prompt_removes_drift_with_enough_points(
    mock_prompt: MagicMock,
    mock_compensate_drift: MagicMock,
    pol_deg: int,
) -> None:
    """Test that drift removal runs when exactly enough points are available.

    The actual drift-removal math is mocked out; this only checks that
    drift_removal_prompt wires the chosen degree through to compensate_drift
    and returns its result.
    """
    mock_prompt.return_value = pol_deg
    compensated_data = MagicMock(name="compensated_data")
    mock_compensate_drift.return_value = compensated_data
    data = _make_data(pol_deg + 1)  # exactly enough points
    result_data, result_deg = drift_removal_prompt(data)
    assert result_data is compensated_data
    assert result_deg == pol_deg
    mock_compensate_drift.assert_called_once_with(pol_deg)
