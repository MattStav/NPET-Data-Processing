import _tkinter
from pathlib import Path

import pytest
from unittest.mock import patch, call

from NPET_DP.epoch_processing.pps import main_pps
from NPET_DP.framework.path_handler import get_plot_path


def test_returns_when_no_files():
    """Test that the function returns when there is no file to process"""
    with patch(
        "NPET_DP.epoch_processing.pps.user_file_select",
        side_effect=FileNotFoundError,
    ):
        main_pps()


@pytest.mark.flaky(reruns=3, only_rerun=_tkinter.TclError.__name__)
@pytest.mark.parametrize("filename", ["test_PPS.out", "test_PPS_long.out"])
def test_main_pps_creates_a_plot(data_dir: Path, tmp_path: Path, filename: str):
    """Test that the main function successfully creates a plot and saves it to the correct path."""
    test_file: Path = data_dir / filename
    with (
        patch(
            "NPET_DP.epoch_processing.pps.user_file_select",
            return_value=test_file,
        ),
        patch("NPET_DP.framework.path_handler.get_path", return_value=tmp_path),
        patch("NPET_DP.epoch_processing.pps.plt.show") as mock_show,
        patch(
            "NPET_DP.epoch_processing.pps.drift_removal_prompt",
            side_effect=lambda data: (data, 0),
        ),
    ):
        main_pps()
        assert get_plot_path(f"pps_{test_file.stem}").is_file()
        assert get_plot_path(f"tdev_{test_file.stem}").is_file()
    assert mock_show.call_count == 2
    mock_show.assert_has_calls([call(block=False), call(block=False)])
