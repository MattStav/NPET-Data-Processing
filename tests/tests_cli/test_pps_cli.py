import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from NPET_DP.main_cli import npet_dp


@pytest.mark.parametrize(
    "test_file, expected_mean, expected_mean_unit, expected_std, expected_std_unit",
    [
        ("test_PPS.out", -5.67741, "ps", 18.2123, "ps"),
        ("test_PPS_long.out", -36.52495, "ps", 26.8100, "ps"),
        ("test_PPS_nonzero.out", -498.09178, "ms", 24.1224, "ps"),
    ],
)
def test_pps_cli_arrives_at_correct_results(
    data_dir: Path,
    tmp_path: Path,
    test_file: str,
    expected_mean: float,
    expected_mean_unit: str,
    expected_std: float,
    expected_std_unit: str,
) -> None:
    """Test the PPS workflow through the actual CLI.

    Drives the app the same way a user would: launches the CLI, picks the
    PPS menu option and answers the drift-removal prompt, then checks the
    printed mean/STD match the values established by the raw workflow test.
    """
    input_dir: Path = tmp_path / "data"
    input_dir.mkdir()
    shutil.copy(data_dir / test_file, input_dir / test_file)
    plots_dir: Path = tmp_path / "plots"
    plots_dir.mkdir()
    name: str = Path(test_file).stem

    runner = CliRunner()
    # Menu: 3 (PPS) -> file is auto-selected (only one present) -> "0" no drift -> 0 (exit menu)
    user_input = "3\n0\n0\n"
    with (
        patch("NPET_DP.workflows.pps.plt.show"),
        patch("NPET_DP.framework.path_handler.get_path", return_value=plots_dir),
    ):
        result = runner.invoke(
            npet_dp,
            ["--data-path", str(input_dir)],
            input=user_input,
        )

    assert result.exit_code == 0, result.output
    assert (
        f"{name} mean delay = {expected_mean:.5f} {expected_mean_unit}" in result.output
    )
    assert f"{name} STD = {expected_std:.4f} {expected_std_unit}" in result.output
