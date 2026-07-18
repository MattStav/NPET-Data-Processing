from functools import cache, wraps
from inspect import Signature, signature
from pathlib import Path
from typing import Callable, Literal, Optional, get_args

import numpy as np
import typer
from numpy.typing import NDArray

_UNITS_TYPE = Literal["s", "ms", "us", "ns", "ps", "fs"]
_UNITS_SCALE: tuple[_UNITS_TYPE] = get_args(_UNITS_TYPE)
DATA_TYPE = [("seconds", np.int_), ("femto", np.int_)]


def validate_inputs(func: Callable) -> Callable:
    """
    Decorator that validates any argument of the decorated function whose name starts with 'data',
    regardless of whether it's passed positionally or as a keyword argument.
    """
    sig: Signature = signature(func)
    # Find all parameter names starting with "data"
    data_params: list[str] = [p for p in sig.parameters if p.startswith("data")]
    if not data_params:
        func_name = getattr(func, "__name__", repr(func))
        raise TypeError(f"Expected 'data' argument for {func_name}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name in data_params:
            if name not in bound.arguments:
                continue  # e.g., optional param not supplied
            value = bound.arguments[name]
            check_data_structure(value, arg_name=name)
        return func(*args, **kwargs)

    return wrapper


def check_data_structure(data: NDArray, arg_name: Optional[str] = None) -> None:
    """
    Check if the data has the correct structure
    :param data: Data to check.
    :param arg_name: The name of the argument being checked (for error messages)
    :raises ValueError: If the data is not in the correct format
    """
    name: str = arg_name or "data"
    if not data.ndim == 1:
        raise ValueError(f"'{name}' must be 1D")
    if data.dtype != DATA_TYPE:
        raise ValueError(f"'{name}' missing fields: 'seconds, femto'")


def import_data(path: Path, seconds_add: Optional[int] = None) -> NDArray:
    """
    Import data from the epoch output files,
    which should be in the format of: `int_sec frac_sec`.
    Basic preprocessing is applied to the data,
    including converting the fractional part to femtoseconds and handling overflow.
    :param path: Path to the epoch output file
    :param seconds_add: Number of seconds to add to each epoch, can be positive or negative
    :return: Array of data in the format exported by this FW
    """
    assert path.is_file(), f"File {path} does not exist"
    assert path.suffix == ".out", f"File {path} is not an epoch output file"
    typer.echo(f"Importing data from {path}")
    data = np.loadtxt(path, dtype=str)
    seconds: NDArray[np.int_] = data[:, 0].astype(int)
    frac_part: NDArray[np.str_] = data[:, 1].astype(str)
    # Overflow femto 1.0 measurement into the seconds
    overflow_mask: NDArray[np.bool_] = np.char.startswith(frac_part, "1.")
    frac_part[overflow_mask] = "0.0"
    seconds[overflow_mask] += 1
    # Add optional seconds offset
    if seconds_add is not None:
        seconds += seconds_add
    # Convert to femtoseconds and remove the decimal part
    femto = np.array([int(v.replace("0.", "").ljust(15, "0")) for v in frac_part])
    data: NDArray = np.array(list(zip(seconds, femto, strict=False)), dtype=DATA_TYPE)
    check_data_structure(data)
    return data


def auto_scale_data(
    data: NDArray[np.int_ | np.floating],
    max_scale: Optional[int] = None,
) -> tuple[NDArray[np.floating], int]:
    """
    Scale a single column (up or down) until the data is in nice format. Less than 1000 more than 1.
    :param data: Data to scale, single column
    :param max_scale: Maximum number of times to scale the data
    :return: Scaled data and the number of times the data was scaled,
    positive scale_iter means upscaling, negative scale_iter means downscaling
    """
    assert data.ndim == 1, "Data must be 1D"
    assert max_scale is None or max_scale >= 0, "Max scale must be positive"
    scale_iter: int = 0
    while True:
        if max_scale is not None and abs(scale_iter) >= max_scale:
            break
        scaled_data = data * (1000**scale_iter)
        if np.abs(scaled_data.max()) < 1:
            scale_iter += 1
        elif np.abs(scaled_data.max()) > 1000:
            scale_iter -= 1
        else:
            break
    return (data * (1000**scale_iter)).astype(np.float64), scale_iter


def scale_data(
    data: NDArray[np.int_ | np.floating],
    scale_power: int,
) -> NDArray[np.float64]:
    """
    Scale a single column (up or down) by a given power of 1000.
    :param data: Data to scale, single column
    :param scale_power: Power of 1000 to scale the data by
    :return: Scaled data
    """
    assert data.ndim == 1, "Data must be 1D"
    return (data * (1000**scale_power)).astype(np.float64)


@cache
def auto_scale_num(
    num: int | float | np.floating,
    max_scale: Optional[int] = None,
) -> tuple[float | np.floating, int]:
    """
    Scale a single number (up or down) until it is in nice format. Less than 1000 more than 1.
    :param num: Number to scale
    :param max_scale: Maximum number of times to scale the number
    :return: Scaled number and the number of times the number was scaled,
    positive scale_iter means upscaling, negative scale_iter means downscaling
    """
    assert isinstance(num, (int, float, np.floating)), "Number must be int or float"
    assert max_scale is None or max_scale >= 0, "Max scale must be positive"
    scale_iter: int = 0
    while True:
        if max_scale is not None and abs(scale_iter) >= max_scale:
            break
        scaled_num = num * (1000**scale_iter)
        if np.abs(scaled_num) < 1:
            scale_iter += 1
        elif np.abs(scaled_num) > 1000:
            scale_iter -= 1
        else:
            break
    return (num * (1000**scale_iter)), scale_iter


def scale_num(num: int | float | np.floating, scale_power: int) -> float | np.floating:
    """
    Scale a single number (up or down) by a given power of 1000.
    :param num: Number to scale
    :param scale_power: Power of 1000 to scale the number by
    :return: Scaled number
    """
    assert isinstance(num, (int, float, np.floating)), "Number must be int or float"
    return num * (1000**scale_power)


def get_unit(original_unit: _UNITS_TYPE, scale_iter: int) -> _UNITS_TYPE:
    """
    Get the unit resulting from scaling data by scale_iter steps of 1000 away from original_unit,
    as produced by auto_scale_data/auto_scale_num (positive scale_iter moves to a finer unit, e.g. s -> ms).
    :param original_unit: Starting unit
    :param scale_iter: Number of 1000x scale steps applied
    :return: Resulting unit
    :raises ValueError: If the resulting unit falls outside the supported range
    """
    index: int = _UNITS_SCALE.index(original_unit) + scale_iter
    if not 0 <= index < len(_UNITS_SCALE):
        raise ValueError(f"Out of supported range {_UNITS_SCALE}")
    return _UNITS_SCALE[index]
