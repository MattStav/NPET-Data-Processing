import shutil
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from NPET_DP.framework.config import config
from NPET_DP.main_cli import npet_dp


def test_two_epochs_cli_arrives_at_correct_results(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """Test the TWO EPOCHS workflow through the actual CLI.

    Drives the app the same way a user would: launches the CLI, picks the
    dual-epoch menu option, selects the START/STOP files and answers the
    following prompts, then checks the printed results match the values
    established by the raw workflow test.
    """
    input_dir: Path = tmp_path / "data"
    input_dir.mkdir()
    shutil.copy(data_dir / "test_data_START.out", input_dir / "test_data_START.out")
    shutil.copy(data_dir / "test_data_STOP.out", input_dir / "test_data_STOP.out")
    plots_dir: Path = tmp_path / "plots"
    plots_dir.mkdir()

    config.frequency = 500
    config.sigma = 2.2

    runner = CliRunner()
    # Menu: 2 (two epochs) -> manually select START file by name -> STOP file
    # is auto-selected (only one left) -> "no" rescale -> "0" no drift -> 0 (exit menu)
    user_input = "2\n0\ntest_data_START.out\nno\n0\n0\n"
    with (
        patch("NPET_DP.workflows.two_epochs.show"),
        patch("NPET_DP.processing.plotting.plt.show"),
        patch("NPET_DP.framework.path_handler.get_path", return_value=plots_dir),
    ):
        result = runner.invoke(
            npet_dp,
            ["--data-path", str(input_dir)],
            input=user_input,
        )

    assert result.exit_code == 0, result.output
    assert "Discarded 7729 epochs from STOP data" in result.output
    assert "Number of accepted values: 32220" in result.output
    assert "Autodetect found a single return signal!" in result.output
    assert "Mean: 59.5264 ns" in result.output
    assert "STD: 17.1517 ps" in result.output
    assert "Accepted values in filtering = 1567" in result.output
    assert "Number of iterations = 18" in result.output
    assert "Return rate: 4.86%" in result.output
