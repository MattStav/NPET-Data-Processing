from unittest.mock import patch
from NPET_DP.framework.settings_menu import settings_menu

@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_frequency(mock_config, mock_prompt):
    """Test that the settings menu prompts the user for the data gathering frequency."""
    mock_prompt.return_value = 1
    settings_menu()
    mock_config.prompt_frequency.assert_called_once()

@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_sigma(mock_config, mock_prompt):
    """Test that the settings menu prompts the user for the sigma value."""
    mock_prompt.return_value = 2
    settings_menu()
    mock_config.prompt_sigma.assert_called_once()

@patch("NPET_DP.framework.settings_menu.typer.prompt")
@patch("NPET_DP.framework.settings_menu.config")
def test_settings_menu_data_dir(mock_config, mock_prompt):
    """Test that the settings menu prompts the user for the data directory."""
    mock_prompt.return_value = 3
    settings_menu()
    assert mock_config.input_data_dir is None

@patch("NPET_DP.framework.settings_menu.typer.prompt")
def test_settings_menu_exit(mock_prompt):
    """Test that the settings menu exits when the user selects option 0."""
    mock_prompt.return_value = 0
    assert settings_menu() is None
