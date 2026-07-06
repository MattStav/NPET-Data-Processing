from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from NPET_DP.main_cli import npet_dp


def test_launching_app_enters_main_menu(tmp_path, monkeypatch):
    """Test that the app enters the main menu and then exits the app when the user selects option 0."""
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    with patch("NPET_DP.main_cli.main_menu") as mock_main_menu:
        result = runner.invoke(npet_dp, input="0\n")
    assert result.exit_code == 0
    mock_main_menu.assert_called_once()


def test_launching_app_with_wrong_data_path_exits_10(tmp_path):
    """Test that the app exits with code 10 when launched with a non-existent data path."""
    runner = CliRunner()
    bad_path = tmp_path / "does_not_exist"
    result = runner.invoke(npet_dp, ["--data-path", str(bad_path)])
    assert result.exit_code == 10


@pytest.mark.parametrize("menu_option, target_function", [
    (1, "NPET_DP.main_cli.main_one_epoch"),
    (2, "NPET_DP.main_cli.main_two_epochs"),
    (3, "NPET_DP.main_cli.main_pps"),
])
def test_app_calls_correct_functions(menu_option, target_function):
    """Test that the app calls the correct functions when launched."""
    runner = CliRunner()
    with patch(target_function) as mock_target:
        runner.invoke(npet_dp, input=f"{menu_option}\n")
        mock_target.assert_called_once()
