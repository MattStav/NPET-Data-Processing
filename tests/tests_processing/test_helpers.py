from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy.typing import NDArray
from pytest import FixtureRequest

from NPET_DP.processing.helpers import (
    _UNITS_TYPE,
    DATA_TYPE,
    auto_scale_data,
    auto_scale_num,
    check_data_structure,
    get_unit,
    import_data,
    scale_data,
    scale_num,
    validate_inputs,
)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((1500, -1), id="once"),
        pytest.param((1500000, -2), id="twice"),
        pytest.param((500, 0), id="pass"),
        pytest.param((33_000_000_000_000_000, -5), id="large"),
    ],
)
def test_num(request: FixtureRequest) -> tuple[float, int]:
    """Fixture providing test cases for number auto-scaling.

    :return: A number and the number of times it should be auto-scaled.
    """
    # noinspection PyUnresolvedReferences
    return request.param


def test_import_data(tmp_path: Path) -> None:
    """Test that the data is correctly imported from a file.

    The data stored in file is in format `123 0.13456`, during import
    it should be correctly process into two int columns: `123 13456`.
    Where the second column represents the decimal values.
    This specific format is chosen to ensure sufficient precision when dealing with 15-digit decimals.
    """
    file_content: list[str] = [
        "100 0.123456789012345",
        "100 1.000000000000000",
        "102 0.999999999999999",
    ]
    test_file: Path = tmp_path / "test_data.out"
    with test_file.open("w") as f:
        f.writelines("\n".join(file_content))
    data: npt.NDArray[np.int_] = import_data(test_file)
    assert len(data) == 3
    assert data[0]["seconds"] == 100
    assert data[0]["femto"] == 123456789012345
    assert data[1]["seconds"] == 101  # 100 + 1 (overflow)
    assert data[1]["femto"] == 0
    assert data[2]["seconds"] == 102
    assert data[2]["femto"] == 999999999999999


@pytest.mark.parametrize("seconds_add", [0, 10, 100, -20])
def test_import_data_with_seconds_add(tmp_path: Path, seconds_add: int) -> None:
    """Test adding seconds during import.

    It is possible to add fixed amount of seconds to the imported data.
    Test that the seconds are correctly added to the timestamps during import.
    """
    file_content = "100 0.123456789012345\n101 0.123456123012345"
    test_file = tmp_path / "test_data_2.out"
    test_file.write_text(file_content)
    data: npt.NDArray[np.int_] = import_data(test_file, seconds_add=seconds_add)
    assert data[0]["seconds"] == 100 + seconds_add
    assert data[0]["femto"] == 123456789012345
    assert data[1]["seconds"] == 101 + seconds_add
    assert data[1]["femto"] == 123456123012345


def test_auto_scale_num(test_num: tuple[float, int]) -> None:
    """Test auto-scaling numbers.

    Test that the number is correctly scaled to a human-readable values.
    This means the number should be 1 < num < 1000
    """
    num, scale = auto_scale_num(test_num[0])
    assert scale == test_num[1]
    assert num == test_num[0] * 1000 ** test_num[1]
    assert 1 < num < 1000


@pytest.mark.parametrize("max_scale", [0, 1, 2])
def test_auto_scale_num_max_scale(max_scale: int, test_num: tuple[float, int]) -> None:
    """Test auto-scaling max_scale.

    Test that auto-scaling respects that max defined scale iter.
    """
    _, scale = auto_scale_num(test_num[0], max_scale=max_scale)
    assert scale == max(test_num[1], -max_scale)


@pytest.mark.parametrize(
    "data, scale",
    ((np.array([10000, 20000, 30000]), -1), (np.array([0.01, 0.02, 0.03]), 1)),
)
def test_auto_scale_data(data: NDArray, scale: int) -> None:
    """Test data auto-scaling.

    Test that the data is correctly scaled.
    This means the data max value should be 1 < data < 1000
    """
    scaled, scale_iter = auto_scale_data(data)
    assert np.array_equal(scaled, np.array([10.0, 20.0, 30.0]))
    assert scale_iter == scale
    assert 1 < np.max(scaled) < 1000


def test_auto_scale_data_max_scale_zero() -> None:
    """Test that data auto-scale respects that max defined scale iter."""
    data: NDArray[np.int64] = np.array([5000, 6000], dtype=np.int64)
    scaled, scale_iter = auto_scale_data(data, max_scale=0)
    assert np.array_equal(scaled, np.array([5000.0, 6000.0]))
    assert scale_iter == 0


@pytest.mark.parametrize(
    "scale_power, expected",
    (
        pytest.param(0, [1500.0, 2500.0], id="no-op"),
        pytest.param(-1, [1.5, 2.5], id="downscale"),
        pytest.param(1, [1_500_000.0, 2_500_000.0], id="upscale"),
    ),
)
def test_scale_data(scale_power: int, expected: list[float]) -> None:
    """Test that scale_data scales the data by the exact requested power of 1000."""
    data = np.array([1500, 2500])
    scaled = scale_data(data, scale_power)
    assert np.array_equal(scaled, np.array(expected))


def test_scale_data_returns_float_array() -> None:
    """Test that scale_data always returns a float array, even for integer input.

    The scaling can divide the numbers, so the output must be float type.
    """
    data: NDArray[np.int64] = np.array([1, 2, 3], dtype=np.int64)
    scaled = scale_data(data, 0)
    assert scaled.dtype == np.float64


@pytest.mark.parametrize(
    "scale_power, expected",
    (
        pytest.param(0, 1500.0, id="no-op"),
        pytest.param(-1, 1.5, id="downscale"),
        pytest.param(1, 1_500_000.0, id="upscale"),
        pytest.param(2, 1_500_000_000.0, id="upscale-twice"),
    ),
)
def test_scale_num(scale_power: int, expected: float) -> None:
    """Test that scale_num scales the number by the exact requested power of 1000."""
    assert scale_num(1500, scale_power) == expected


def test_scale_num_rejects_non_numeric() -> None:
    """Test that scale_num raises an AssertionError for non-numeric input."""
    with pytest.raises(AssertionError):
        # noinspection bad-argument-type
        scale_num("1500", 1)


@pytest.mark.parametrize(
    "original_unit, scale_iter, expected_unit",
    (
        pytest.param("s", 0, "s", id="no-op"),
        pytest.param("s", 1, "ms", id="forward-one"),
        pytest.param("s", 2, "us", id="forward-two"),
        pytest.param("us", 3, "fs", id="forward-to-end"),
        pytest.param("ms", -1, "s", id="backward-one"),
        pytest.param("fs", -5, "s", id="backward-to-start"),
    ),
)
def test_get_unit(
    original_unit: _UNITS_TYPE,
    scale_iter: int,
    expected_unit: _UNITS_TYPE,
) -> None:
    """Unit getting after scaling.

    Test that the new data unit can be correctly retrieved based on the scaling factor.
    """
    assert get_unit(original_unit, scale_iter) == expected_unit


@pytest.mark.parametrize(
    "original_unit, scale_iter",
    (
        pytest.param("fs", 2, id="past-finest"),
        pytest.param("s", -1, id="past-coarsest"),
        pytest.param("fs", 100, id="far-past-finest"),
    ),
)
def test_get_unit_out_of_range(original_unit: _UNITS_TYPE, scale_iter: int) -> None:
    """Test invalid unit.

    Test that a ValueError is raised when the resulting unit is out of the supported range.
    Only certain units are currently supported.
    """
    with pytest.raises(ValueError):
        get_unit(original_unit, scale_iter)


def test_check_data_structure_valid() -> None:
    """Array check valid structure.

    Test that checking array structure works correctly on valid data.
    """
    data = np.array([(1, 2), (3, 4)], dtype=DATA_TYPE)
    check_data_structure(data)


@pytest.mark.parametrize(
    "data, arg_name, expected_message",
    (
        pytest.param(
            np.array([[(1, 2), (3, 4)]], dtype=DATA_TYPE),
            None,
            "'data' must be 1D",
            id="not-1d-default-name",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            None,
            "'data' missing fields: 'seconds, femto'",
            id="wrong-dtype-default-name",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            "my_arg",
            "'my_arg' must be 1D",
            id="not-1d-custom-name",
        ),
        pytest.param(
            np.array([1, 2, 3]),
            "my_arg",
            "'my_arg' missing fields: 'seconds, femto'",
            id="wrong-dtype-custom-name",
        ),
    ),
)
def test_check_data_structure_invalid(
    data: NDArray,
    arg_name: str,
    expected_message: str,
) -> None:
    """Array check valid structure on invalid data.

    Test that checking array structure works correctly on invalid data.
    Should raise ValueError when invalid data is passed in.
    """
    with pytest.raises(ValueError, match=expected_message):
        check_data_structure(data, arg_name=arg_name)


def test_validate_inputs_requires_data_param() -> None:
    """Validate inputs only checks data params.

    Test that decorating a function without a 'data...' parameter raises a TypeError.
    """
    with pytest.raises(TypeError, match="Expected 'data' argument"):

        @validate_inputs
        def func(x: Any) -> Any:
            return x


@pytest.mark.parametrize(
    "call",
    (
        pytest.param(lambda func, data: func(data), id="positional"),
        pytest.param(lambda func, data: func(data=data), id="keyword"),
    ),
)
def test_validate_inputs_passes_valid_data(call: Callable) -> None:
    """Validate input for valid data.

    Test that a function decorated with validate_inputs runs normally for valid data.
    """

    @validate_inputs
    def func(data_test: Any) -> Any:
        return len(data_test)

    data = np.array([(1, 2), (3, 4)], dtype=DATA_TYPE)
    assert call(func, data) == 2


@pytest.mark.parametrize(
    "data, expected_message",
    (
        pytest.param(np.array([[1, 2]]), "must be 1D", id="not-1d"),
        pytest.param(np.array([1, 2, 3]), "missing fields", id="wrong-dtype"),
    ),
)
@pytest.mark.parametrize(
    "call",
    (
        pytest.param(lambda func, data: func(data), id="positional"),
        pytest.param(lambda func, data: func(data=data), id="keyword"),
    ),
)
def test_validate_inputs_rejects_invalid_data(
    call: Callable,
    data: NDArray,
    expected_message: str,
) -> None:
    """Validate inputs for invalid data.

    Test that a function decorated with validate_inputs raises ValueError when invalid data is passed in.
    """

    @validate_inputs
    def func(data_test: Any) -> Any:
        return data_test

    with pytest.raises(ValueError, match=expected_message):
        call(func, data)


@pytest.mark.parametrize(
    "invalid_param",
    (
        pytest.param("data_a", id="first-param-invalid"),
        pytest.param("data_b", id="second-param-invalid"),
    ),
)
def test_validate_inputs_checks_every_data_prefixed_param(invalid_param: str) -> None:
    """Test validate inputs with multiple arguments.

    Test that validate_inputs validates every parameter whose name starts with 'data'.
    """

    @validate_inputs
    def func(data_a: Any, data_b: Any) -> Any:
        return data_a, data_b

    valid = np.array([(1, 2)], dtype=DATA_TYPE)
    invalid = np.array([1, 2, 3])
    values = {invalid_param: invalid}
    kwargs = {
        "data_a": values.get("data_a", valid),
        "data_b": values.get("data_b", valid),
    }
    with pytest.raises(ValueError, match=f"'{invalid_param}' missing fields"):
        func(**kwargs)


def test_validate_inputs_ignores_non_data_params() -> None:
    """Test that validate_inputs does not validate parameters not prefixed with 'data'."""

    @validate_inputs
    def func(other: Any, data: Any) -> Any:
        return other, data

    valid = np.array([(1, 2)], dtype=DATA_TYPE)
    assert func(other="not an array", data=valid) == ("not an array", valid)
