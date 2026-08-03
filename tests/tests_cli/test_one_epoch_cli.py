import shutil
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from NPET_DP.framework.config import config
from NPET_DP.framework.constants import FEMTO
from NPET_DP.main_cli import npet_dp
from NPET_DP.processing.data_struct import NPETData


def test_one_epoch_cli_arrives_at_correct_results(
    data_dir: Path,
    tmp_path: Path,
) -> None:
    """Test the ONE EPOCH workflow through the actual CLI.

    Drives the app the same way a user would: launches the CLI, picks the
    single-epoch menu option and selects the file, then checks the modulo
    delay data reaching the plot matches the same computation done directly
    against the NPETData structure. At the fixed frequency of 500 Hz, this
    test data produces no auto-detectable signal, so the histogram prompt is
    declined.
    """
    test_file: str = "test_data_STOP.out"
    input_dir: Path = tmp_path / "data"
    input_dir.mkdir()
    shutil.copy(data_dir / test_file, input_dir / test_file)
    plots_dir: Path = tmp_path / "plots"
    plots_dir.mkdir()

    config.frequency = 500
    expected_mod_data: NPETData = NPETData.from_path(data_dir / test_file).modulo(
        round(FEMTO / config.frequency)
    )

    runner = CliRunner()
    # Menu: 1 (one epoch) -> file is auto-selected (only one present) ->
    # "n" to decline the histogram prompt (no signal was auto-detected) -> 0 (exit menu)
    user_input = "1\nn\n0\n"
    with (
        patch("NPET_DP.workflows.one_epoch.plt.show"),
        patch("NPET_DP.framework.path_handler.get_path", return_value=plots_dir),
        patch(
            "NPET_DP.workflows.one_epoch.__plot_singular_data"
        ) as mock_plot_singular,
    ):
        result = runner.invoke(
            npet_dp,
            ["--data-path", str(input_dir)],
            input=user_input,
        )

    assert result.exit_code == 0, result.output
    assert "Unable to autodetect a single signal" in result.output
    mock_plot_singular.assert_called_once()
    plotted_data: NPETData = mock_plot_singular.call_args.args[0]
    assert plotted_data.femto.tolist() == expected_mod_data.femto.tolist()
    assert plotted_data.seconds.tolist() == expected_mod_data.seconds.tolist()
