from unittest.mock import MagicMock, patch

import pytest
from pytest import CaptureFixture

from NPET_DP.framework.settings_menu import settings_menu


@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_frequency(
    mock_config: MagicMock,
    mock_prompt: MagicMock,
) -> None:
    """Test settings menu frequency.

    Test that the settings can prompt the user for the data gathering frequency.
    """
    mock_prompt.return_value = 1
    settings_menu()
    mock_config.prompt_frequency.assert_called_once()


@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_sigma(
    mock_config: MagicMock,
    mock_prompt: MagicMock,
) -> None:
    """Test settings menu sigma.

    Test that the settings menu can prompt the user for the sigma value.
    """
    mock_prompt.return_value = 2
    settings_menu()
    mock_config.prompt_sigma.assert_called_once()


@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_data_dir(
    mock_config: MagicMock,
    mock_prompt: MagicMock,
) -> None:
    """Test settings menu data dir.

    Test that the settings menu can prompt the user for the data directory.
    """
    mock_prompt.return_value = 3
    settings_menu()
    assert mock_config.input_data_dir is None


@patch("NPET_DP.framework.settings_menu.typer.prompt")
def test_settings_menu_exit(mock_prompt: MagicMock) -> None:
    """Test that the settings menu exits when the user selects option 0."""
    mock_prompt.return_value = 0
    assert settings_menu() is None


@pytest.mark.parametrize("invalid_choice", [-1, 4, 99])
@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_invalid_choice(
    mock_config: MagicMock,
    mock_prompt: MagicMock,
    capsys: CaptureFixture,
    invalid_choice: int,
) -> None:
    """Test settings menu invalid choice.

    Test that an unrecognized choice hits the `case _` fallback: it prints an
    error, returns without prompting further, and leaves config untouched.
    """
    mock_prompt.return_value = invalid_choice
    assert settings_menu() is None
    assert "Invalid choice" in capsys.readouterr().out
    mock_config.prompt_frequency.assert_not_called()
    mock_config.prompt_sigma.assert_not_called()
