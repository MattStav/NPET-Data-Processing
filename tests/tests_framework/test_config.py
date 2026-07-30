from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest

from NPET_DP.framework.config import __AppConfig


def test_config_default_data_dir() -> None:
    """Test that the default data directory is the CWD."""
    config = __AppConfig()
    assert config.input_data_dir == Path.cwd()


def test_input_data_dir_setter_accepts_existing_path(tmp_path: Path) -> None:
    """Test valid data dir setting.

    Test that the data directory can be set to an existing path.
    Failing this test signals an error in directory validation logic.
    """
    config = __AppConfig()
    config.input_data_dir = tmp_path
    assert config.input_data_dir == tmp_path
    assert config.input_data_dir.is_dir(), "Directory does not exist"
    assert config.input_data_dir.is_absolute(), "The Path should be absolute"


def test_input_data_dir_setter_prompts(tmp_path: Path) -> None:
    """Test data directory user prompt.

    Test that the data directory can be set to None,
    which leads to the user being prompted to define new directory.
    Failing this test means signals error in the data directory setting logic.
    """
    config = __AppConfig()
    with patch("typer.prompt", return_value=str(tmp_path)):
        config.input_data_dir = None
    assert config.input_data_dir == tmp_path


def test_input_data_dir_prompt_accepts_relative_path(tmp_path: Path) -> None:
    """Test data directory accepts relative path.

    Test that the data directory prompt accepts a relative path,
    in which case the path should be appended to the latest selection.
    """
    config = __AppConfig()
    config.input_data_dir = tmp_path
    some_dir: Path = tmp_path / "some_dir"
    some_dir.mkdir()
    with patch("typer.prompt", return_value=f"./{some_dir.name}"):
        config.input_data_dir = None
    assert config.input_data_dir == some_dir
    assert config.input_data_dir.is_dir(), "Directory does not exist"
    assert config.input_data_dir.is_absolute(), "The Path should be absolute"
    assert config.input_data_dir.is_relative_to(tmp_path), "Path not relative"


def test_app_config_invalid_data_dir() -> None:
    """Test invalid data directory.

    Test that an error is raised if the data directory does not exist.
    """
    config = __AppConfig()
    with pytest.raises(FileNotFoundError):
        config.input_data_dir = Path("/non/existent/path/at/least/unlikely")


@pytest.mark.parametrize("freq", [100, 1000, 534])
def test_app_config_frequency(freq: int) -> None:
    """Test that the frequency can be set to a valid value."""
    config = __AppConfig()
    config.frequency = freq
    assert config.frequency == freq


@pytest.mark.parametrize("freq", [0, -1, "a"])
def test_app_config_frequency_invalid(freq: Any) -> None:
    """Test invalid frequency.

    Test that an error is raised if the frequency is invalid.
    """
    config = __AppConfig()
    with pytest.raises(ValueError):
        config.frequency = freq


@pytest.mark.parametrize("sigma", [2.0, 2.3, 3.0])
def test_app_config_sigma(sigma: float) -> None:
    """Test that the sigma can be set to a valid value."""
    config = __AppConfig()
    config.sigma = sigma
    assert config.sigma == sigma


@pytest.mark.parametrize("sigma", [0.0, -1.1, "a"])
def test_app_config_sigma_invalid(sigma: Any) -> None:
    """Test invalid sigma.

    Test that an error is raised if the sigma is invalid.
    """
    config = __AppConfig()
    with pytest.raises(ValueError):
        config.sigma = sigma


def test_app_config_sigma_default_value() -> None:
    """Test the default sigma value.

    The default sigma should be 2.2,
    the most commonly used values for processing single photon detector's data.
    """
    config = __AppConfig()
    with patch("typer.prompt", return_value=3.0) as mock_prompt:
        config.prompt_sigma()
    assert mock_prompt.call_args.kwargs["default"] == 2.2


@pytest.mark.parametrize("min_delay", [100.0, 1000.23, 534.01])
@pytest.mark.parametrize("max_delay", [200.0, 2000.48, 12.3])
def test_app_config_delays_valid(min_delay: float, max_delay: float) -> None:
    """Test that the min and max delays can be set to valid values."""
    config = __AppConfig()
    config.min_delay = min_delay
    assert config.min_delay == min_delay
    config.max_delay = max_delay
    assert config.max_delay == max_delay


@patch("typer.prompt")
def test_app_config_prompt_frequency(mock_prompt: MagicMock) -> None:
    """Test frequency prompt.

    Test that the frequency can be set using the prompt.
    """
    config = __AppConfig()
    mock_prompt.return_value = 50
    config.prompt_frequency()
    assert config.frequency == 50


@patch("typer.prompt")
def test_app_config_prompt_sigma(mock_prompt: MagicMock) -> None:
    """Test sigma prompt.

    Test that the sigma can be set using the prompt.
    """
    config = __AppConfig()
    mock_prompt.return_value = 1.5
    config.prompt_sigma()
    assert config.sigma == 1.5


@pytest.mark.parametrize(
    "delay_type, input_val, initial_min, initial_max, expected_val",
    [
        ("min", 0.5, None, None, 0.5 * 1e6),
        ("max", 2.0, None, None, 2.0 * 1e6),
        # Testing validation against each other
        ("min", 0.5, None, 1.0 * 1e6, 0.5 * 1e6),
        ("max", 2.0, 1.0 * 1e6, None, 2.0 * 1e6),
    ],
)
@patch("typer.prompt")
def test_app_config_prompt_delay_valid(
    mock_prompt: MagicMock,
    delay_type: Literal["min", "max"],
    input_val: float,
    initial_min: float | None,
    initial_max: float | None,
    expected_val: float,
) -> None:
    """Test valid delay prompts.

    The prompt accepts values in ns, but the storage is in fs!
    """
    config = __AppConfig()
    if initial_min is not None:
        config.min_delay = initial_min
    if initial_max is not None:
        config.max_delay = initial_max
    mock_prompt.return_value = input_val
    config.prompt_delay(delay_type)
    if delay_type == "min":
        assert config.min_delay == expected_val
    else:
        assert config.max_delay == expected_val


@pytest.mark.parametrize(
    "delay_type, inputs, initial_min, initial_max, expected_final_val",
    [
        # Min >= Max
        ("min", [2.0, 0.5], None, 1.0 * 1e6, 0.5 * 1e6),
        # Max <= Min
        ("max", [0.5, 2.0], 1.0 * 1e6, None, 2.0 * 1e6),
    ],
)
@patch("typer.prompt")
def test_app_config_prompt_delay_invalid_retry(
    mock_prompt: MagicMock,
    delay_type: Literal["min", "max"],
    inputs: list[float],
    initial_min: float | None,
    initial_max: float | None,
    expected_final_val: float,
) -> None:
    """Test delay invalid prompt inputs.

    Test that invalid delay values reprompt for until a valid value is provided.
    The prompt should only accept values where the max > min.
    """
    config = __AppConfig()
    if initial_min is not None:
        config.min_delay = initial_min
    if initial_max is not None:
        config.max_delay = initial_max
    mock_prompt.side_effect = inputs
    config.prompt_delay(delay_type)
    if delay_type == "min":
        assert config.min_delay == expected_final_val
    else:
        assert config.max_delay == expected_final_val
    assert mock_prompt.call_count == len(inputs)


@pytest.mark.parametrize(
    "delay_type, input_val, initial_min, initial_max, expected_val",
    [
        ("min", 2.0, None, 1.0 * 1e6, 2.0 * 1e6),
        ("max", 0.5, 1.0 * 1e6, None, 0.5 * 1e6),
    ],
)
@patch("typer.prompt")
def test_app_config_prompt_delay_no_validate(
    mock_prompt: MagicMock,
    delay_type: Literal["min", "max"],
    input_val: float,
    initial_min: float | None,
    initial_max: float | None,
    expected_val: float,
) -> None:
    """Test delay prompt with no validation.

    Test that the delay can be set using the prompt without any validation.
    When validation is disabled the prompt should accept values where max < min.
    """
    config = __AppConfig()
    if initial_min is not None:
        config.min_delay = initial_min
    if initial_max is not None:
        config.max_delay = initial_max
    mock_prompt.return_value = input_val
    config.prompt_delay(delay_type, validate=False)
    if delay_type == "min":
        assert config.min_delay == expected_val
    else:
        assert config.max_delay == expected_val


@pytest.mark.parametrize(
    "delay_type, initial_val, expected_default",
    [
        ("min", None, 0),
        ("max", None, 0),
        ("min", 1.23 * 1e6, 1.23),
        ("max", 4.56 * 1e6, 4.56),
    ],
)
@patch("typer.prompt")
def test_app_config_prompt_delay_default_value(
    mock_prompt: MagicMock,
    delay_type: Literal["min", "max"],
    initial_val: float | None,
    expected_default: float,
) -> None:
    """Test prompt default value.

    Test that the prompt default value is correctly calculated.
    The default should be zero when no value has been set yet,
    otherwise it should be the current value in ns.
    """
    config = __AppConfig()
    if delay_type == "min":
        if initial_val is not None:
            config.min_delay = initial_val
    else:
        if initial_val is not None:
            config.max_delay = initial_val
    mock_prompt.return_value = 10.0
    config.prompt_delay(delay_type)
    assert mock_prompt.call_args.kwargs["default"] == expected_default
