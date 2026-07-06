from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from NPET_DP.epoch_processing.helper_funcs import (
    auto_scale_data,
    auto_scale_num,
    import_data,
    get_unit,
    scale_data,
    scale_num,
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
def test_num(request) -> tuple[float, int]:
    return request.param


def test_import_data(tmp_path):
    """Test that the data is correctly imported from a file."""
    file_content: list[str] = [
        "100 0.123456789012345",
        "101 1.000000000000000",
        "102 0.999999999999999",
    ]
    test_file: Path = tmp_path / "test_data.out"
    with test_file.open("w") as f:
        f.writelines("\n".join(file_content))
    data: npt.NDArray[np.int_] = import_data(test_file)
    assert len(data) == 3
    assert data[0]["seconds"] == 100
    assert data[0]["femto"] == 123456789012345
    assert data[1]["seconds"] == 102  # 101 + 1 (overflow)
    assert data[1]["femto"] == 0
    assert data[2]["seconds"] == 102
    assert data[2]["femto"] == 999999999999999


@pytest.mark.parametrize("seconds_add", [0, 10, 100, -20])
def test_import_data_with_seconds_add(tmp_path, seconds_add: int):
    """Test that the seconds are correctly added to the timestamps during import."""
    file_content = "100 0.123456789012345\n101 0.123456123012345"
    test_file = tmp_path / "test_data_2.out"
    test_file.write_text(file_content)
    data: npt.NDArray[np.int_] = import_data(test_file, seconds_add=seconds_add)
    assert data[0]["seconds"] == 100 + seconds_add
    assert data[0]["femto"] == 123456789012345
    assert data[1]["seconds"] == 101 + seconds_add
    assert data[1]["femto"] == 123456123012345


def test_auto_scale_num(test_num):
    """Test that the number is correctly scaled down."""
    num, scale = auto_scale_num(test_num[0])
    assert scale == test_num[1]
    assert num == test_num[0] * 1000 ** test_num[1]


@pytest.mark.parametrize("max_scale", [0, 1, 2])
def test_auto_scale_num_max_scale(max_scale, test_num):
    """Test that the number is correctly scaled down up to the maximum scale."""
    num, scale = auto_scale_num(test_num[0], max_scale=max_scale)
    assert scale == max(test_num[1], -max_scale)


@pytest.mark.parametrize(
    "data, scale",
    ((np.array([10000, 20000, 30000]), -1), (np.array([0.01, 0.02, 0.03]), 1)),
)
def test_auto_scale_data(data, scale):
    """Test that the data is correctly scaled down or up"""
    scaled, scale_iter = auto_scale_data(data)
    assert np.array_equal(scaled, np.array([10.0, 20.0, 30.0]))
    assert scale_iter == scale


def test_auto_scale_data_max_scale_zero():
    """Test that the data is not scaled down if max_scale is zero"""
    data = np.array([5000, 6000], dtype=np.int64)
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
def test_scale_data_explicit_power(scale_power, expected):
    """Test that scale_data scales the data by the exact requested power of 1000."""
    data = np.array([1500, 2500])
    scaled = scale_data(data, scale_power)
    assert np.array_equal(scaled, np.array(expected))


def test_scale_data_returns_float_array():
    """Test that scale_data always returns a float array, even for integer input."""
    data = np.array([1, 2, 3], dtype=np.int64)
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
def test_scale_num_explicit_power(scale_power, expected):
    """Test that scale_num scales the number by the exact requested power of 1000."""
    assert scale_num(1500, scale_power) == expected


def test_scale_num_rejects_non_numeric():
    """Test that scale_num raises an AssertionError for non-numeric input."""
    with pytest.raises(AssertionError):
        scale_num("1500", 1)  # ty:ignore[invalid-argument-type]


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
def test_get_unit(original_unit, scale_iter, expected_unit):
    """Test that the unit is correctly shifted by scale_iter steps of 1000."""
    assert get_unit(original_unit, scale_iter) == expected_unit


@pytest.mark.parametrize(
    "original_unit, scale_iter",
    (
        pytest.param("fs", 1, id="past-finest"),
        pytest.param("s", -1, id="past-coarsest"),
        pytest.param("fs", 100, id="far-past-finest"),
    ),
)
def test_get_unit_out_of_range(original_unit, scale_iter):
    """Test that a ValueError is raised when the resulting unit is out of the supported range."""
    with pytest.raises(ValueError):
        get_unit(original_unit, scale_iter)
