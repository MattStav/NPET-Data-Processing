from pathlib import Path
from unittest.mock import patch

from NPET_DP.workflows.one_epoch import main_one_epoch
from NPET_DP.framework.config import config
from NPET_DP.framework.path_handler import get_plot_path


def test_returns_when_no_files():
    """Test that the function returns when there is no file to process"""
    with (
        patch(
            "NPET_DP.workflows.one_epoch.user_file_select",
            side_effect=FileNotFoundError,
        ),
        patch("NPET_DP.workflows.one_epoch.NPETData.from_path") as mock_from_path,
        patch("NPET_DP.workflows.one_epoch.__plot_singular_data") as mock_plot,
    ):
        main_one_epoch()
    mock_from_path.assert_not_called()
    mock_plot.assert_not_called()


def test_main_one_epoch_creates_a_plot(data_dir: Path, tmp_path: Path):
    """Test that the main function successfully creates a plot and saves it to the correct path."""
    test_file: Path = data_dir / "test_data_STOP.out"
    config.frequency = 500  # Preset the frequency to avoid user input during the test
    with (
        patch(
            "NPET_DP.workflows.one_epoch.user_file_select",
            return_value=test_file,
        ),
        patch("NPET_DP.framework.path_handler.get_path", return_value=tmp_path),
        patch("NPET_DP.workflows.one_epoch.plt.show") as mock_show,
        patch(
            "NPET_DP.workflows.one_epoch.typer.confirm", return_value=False
        ) as mock_confirm,
    ):
        main_one_epoch()
        assert get_plot_path(f"single_{test_file.stem}").is_file()
    mock_show.assert_called_once_with(block=False)
    mock_confirm.assert_called_once_with("Do you want to plot histogram?")
