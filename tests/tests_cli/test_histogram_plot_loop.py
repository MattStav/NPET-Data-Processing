from unittest.mock import MagicMock, patch

import numpy as np
from pytest import CaptureFixture

from NPET_DP.processing.data_struct import NPETData
from NPET_DP.workflows.helpers import histogram_plot_loop


def _make_delay_data() -> NPETData:
    """Build a tightly clustered NPETData sample.

    All points sit close enough together to fall within a single sigma-filter
    pass and a single detect_signal bin, keeping the loop's math deterministic.
    """
    return NPETData(
        seconds=np.arange(10, dtype=np.int_).astype(np.int_),
        femto=np.array([995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004]),
    )


@patch("NPET_DP.workflows.helpers.plot_histogram")
@patch("NPET_DP.workflows.helpers.config")
@patch("typer.prompt")
def test_histogram_plot_loop_no_rescale_returns_after_one_pass(
    mock_prompt: MagicMock,
    mock_config: MagicMock,
    mock_plot_histogram: MagicMock,
) -> None:
    """Test that choosing "no" rescale runs the loop exactly once."""
    mock_config.sigma = 3.0
    mock_config.min_delay = 500.0
    mock_config.max_delay = 2000.0
    mock_prompt.return_value = "no"
    data = _make_delay_data()
    result = histogram_plot_loop(data, "test_name")
    assert np.array_equal(np.sort(result.femto), data.femto)
    mock_prompt.assert_called_once()
    mock_plot_histogram.assert_called_once()
    assert mock_plot_histogram.call_args.kwargs["name"] == "test_name"


@patch("NPET_DP.workflows.helpers.plot_histogram")
@patch("NPET_DP.workflows.helpers.config")
@patch("typer.prompt")
def test_histogram_plot_loop_manual_rescale_loops_again(
    mock_prompt: MagicMock,
    mock_config: MagicMock,
    mock_plot_histogram: MagicMock,
) -> None:
    """Test that "manual" rescale re-prompts for delay bounds and loops again.

    The actual delay prompting (config.prompt_delay) is mocked out, since it is
    its own separately tested piece of user interaction.
    """
    mock_config.sigma = 3.0
    mock_config.min_delay = 500.0
    mock_config.max_delay = 2000.0
    mock_prompt.side_effect = ["manual", "no"]
    data = _make_delay_data()
    histogram_plot_loop(data, "test_name")
    assert mock_prompt.call_count == 2
    mock_config.prompt_delay.assert_any_call("min", validate=False)
    mock_config.prompt_delay.assert_any_call("max")
    assert mock_plot_histogram.call_count == 2


@patch("NPET_DP.workflows.helpers.rough_auto_range")
@patch("NPET_DP.workflows.helpers.plot_histogram")
@patch("NPET_DP.workflows.helpers.config")
@patch("typer.prompt")
def test_histogram_plot_loop_auto_rescale_with_single_signal_calls_auto_range(
    mock_prompt: MagicMock,
    mock_config: MagicMock,
    mock_plot_histogram: MagicMock,
    mock_auto_range: MagicMock,
) -> None:
    """Test that "auto" rescale calls auto_range when exactly one signal is found.

    The tightly clustered sample data forms a single detect_signal group,
    so autodetection should succeed and hand off to auto_range.
    """
    mock_config.sigma = 3.0
    mock_config.min_delay = 500.0
    mock_config.max_delay = 2000.0
    mock_prompt.side_effect = ["auto", "no"]
    data = _make_delay_data()
    histogram_plot_loop(data, "test_name")
    mock_auto_range.assert_called_once()
    selection_arg, mask_arg = mock_auto_range.call_args.args
    assert np.array_equal(selection_arg.femto, data.femto)
    assert mask_arg.all()
    assert mock_plot_histogram.call_count == 2


@patch("NPET_DP.workflows.helpers.rough_auto_range")
@patch("NPET_DP.workflows.helpers.plot_histogram")
@patch("NPET_DP.workflows.helpers.config")
@patch("typer.prompt")
def test_histogram_plot_loop_auto_rescale_without_single_signal_reprompts(
    mock_prompt: MagicMock,
    mock_config: MagicMock,
    mock_plot_histogram: MagicMock,
    mock_auto_range: MagicMock,
    capsys: CaptureFixture,
) -> None:
    """Test that "auto" rescale skips auto_range when detection isn't a single signal.

    Two widely separated clusters produce two detect_signal groups, so
    autodetection should fail and the loop should print a warning and continue.
    """
    mock_config.sigma = 3.0
    mock_config.min_delay = 1.0
    mock_config.max_delay = 10_000_000.0
    mock_prompt.side_effect = ["auto", "no"]
    data = NPETData(
        seconds=np.arange(10, dtype=np.int_).astype(np.int_),
        femto=np.array(
            [
                1000,
                1001,
                1002,
                999,
                998,
                5_000_000,
                5_000_001,
                5_000_002,
                4_999_999,
                4_999_998,
            ]
        ),
    )
    histogram_plot_loop(data, "test_name")
    mock_auto_range.assert_not_called()
    assert "Failed to autodetect a single signal!" in capsys.readouterr().out
    assert mock_plot_histogram.call_count == 2
