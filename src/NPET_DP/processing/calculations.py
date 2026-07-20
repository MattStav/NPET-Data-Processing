import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from numpy.typing import NDArray

from NPET_DP.processing.helpers import (
    DATA_TYPE,
    check_data_structure,
    validate_inputs,
)
from NPET_DP.framework.constants import FEMTO


@validate_inputs
def process_overflow(data: NDArray) -> NDArray:
    """
    Processes the data overflowing into the previous second.
    Data where the overflow happened is adjusted to match the next second.
    :param data: Data to be processed, in the FW standard format.
    :return: Processed data, in the FW standard format.
    """
    # Difference between each row and the next for the first col
    diffs = np.diff(
        data["seconds"],
        prepend=round(data["seconds"][0] + data["femto"][0] / FEMTO) - 1,
    )
    # Calculate whether overflow happened and adjust the femto value if necessary
    offset = np.zeros(len(data), dtype=np.int64)
    current_offset = 0
    for i, d in enumerate(diffs):
        if d == 0:  # duplicate → start adding 1
            current_offset = FEMTO
        elif d > 1:  # skip → reset offset
            current_offset = 0
        offset[i] = current_offset
    # Adjust for overflow
    ret: NDArray = np.array(
        list(
            zip(
                data["seconds"] + (offset != 0).astype(np.int64),
                data["femto"] - offset,
                strict=True,
            )
        ),
        dtype=DATA_TYPE,
    )
    check_data_structure(ret)
    return ret


def is_continuous(data: NDArray[np.int_], expected_diff: int = 1) -> bool:
    """
    Check whether one column data is continuous, i.e., all values are consecutive.
    :param data: Data to check, in one column.
    :param expected_diff: Expected difference between consecutive values.
    :return: True if the data is continuous, False otherwise.
    """
    assert data.ndim == 1, "Data must be 1D"
    return bool(np.all(np.diff(data, axis=0) == expected_diff))


@validate_inputs
def discard_rows_until_first_col_match(
    *,
    data_ref: NDArray,
    data_to_process: NDArray,
) -> tuple[NDArray, int]:
    """
    Discard rows until the first column of the data matches the first column of the reference data.
    :param data_ref: Reference data.
    :param data_to_process: Data starting too early, where the early part will be discarded.
    :return: Matching portion of the data and the number of discarded rows.
    :raises IndexError: If the data to process is empty or if no match is found.
    """
    discarded: int = 0
    while data_to_process[discarded][0] < data_ref[0][0]:
        discarded += 1
    return data_to_process[discarded:], discarded


@validate_inputs
def calculate_delay(
    *,
    data_start: NDArray,
    data_stop: NDArray,
    frequency: int,
) -> NDArray:
    """
    Calculate the delay between two sets of EPOCH measurement data.
    For each epoch in the first dataset, find the corresponding epoch in the second dataset,
    within the allowed delay range defined by the frequency.
    The delay is calculated as the difference in femtoseconds between the two epochs, taking into account potential overflows.
    If no corresponding epoch is found within the allowed delay range, the epoch from the start dataset is discarded.
    The resulting array contains the calculated delays for all matched epochs.
    :param data_start: First dataset, in the FW standard format.
    :param data_stop: Second dataset, in the FW standard format.
    :param frequency: Data gathering (measurement) frequency.
    :return: Calculated delays in femtoseconds.
    """
    searched_interval_size: float = FEMTO / (2 * frequency)
    # pre-allocated array for the calculated delays
    diff_max_len: int = max(len(data_start), len(data_stop))
    diff_mat: NDArray = np.ones((diff_max_len, 2), dtype=np.int64)
    diff_mat = unstructured_to_structured(diff_mat, dtype=np.dtype(DATA_TYPE))
    index: int = 0
    for epoch_start in data_start:
        for i, epoch_stop in enumerate(data_stop):
            # If no match was found in one measurement interval plus some margin,
            # break and proceed to the next start epoch. The current start epoch is discarded.
            if i == frequency * 2 + 10:
                break
            # Case when the measurements happened on the same second
            if epoch_stop["seconds"] == epoch_start["seconds"]:
                diff: int = epoch_stop["femto"] - epoch_start["femto"]
                diff_abs: int = np.abs(diff)
            # Case when measurements happened on a different second, but they fulfill conditions for potential cross
            elif all(c > s for c, s in zip(epoch_stop, epoch_start, strict=False)) and (
                epoch_stop["femto"] > FEMTO / 2 or epoch_start["femto"] > FEMTO / 2
            ):
                diff_abs = int(
                    FEMTO - np.abs(epoch_stop["femto"] - epoch_start["femto"])
                )
                # Choose the sign of the difference based on which epoch is earlier
                diff: int = (
                    -diff_abs
                    if epoch_stop["seconds"] < epoch_start["seconds"]
                    else diff_abs
                )
            # Case when there is no chance of matching the epochs
            else:
                continue
            # If the difference is bigger than the possible detection interval,
            # skip this combination and proceed to the next stop epoch
            if diff_abs > searched_interval_size:
                continue
            diff_mat[index] = (epoch_start["seconds"], diff)
            index += 1
            # Don't process the same epochs again, slice only the data after the current epoch.
            # The measurements can't travel back in time :)
            data_stop = data_stop[i + 1 :]
            # Break to proceed to the next measurement set
            break
    check_data_structure(diff_mat)
    return diff_mat[:index]  # Trim the array to the actual size


def detect_signal(
    data_delay: NDArray[np.int_],
    *,
    bin_size: int = 40_000,  # 0.04 ns
    percentage_threshold: float = 0.15,
) -> tuple[NDArray[np.bool_], ...]:
    """
    Detect signals in the delay data by identifying horizontal lines (clusters of similar delay values).
    :param data_delay: Data to be processed, the femtoseconds delay column from the data.
    :param bin_size: The size of the bins in femtoseconds into which the data will be split (keyword-only).
    :param percentage_threshold: The percentage of data that must be in a bin to be considered a signal (keyword-only).
    :return: Boolean masks indicating detected signals.
    Each mask corresponds to a detected signal (horizontal line) in the data.
    """
    assert data_delay.ndim == 1, "Data must be 1D"
    assert bin_size > 0, "Bin size must be positive"
    assert 0 <= percentage_threshold <= 10, "Percentage threshold must be [0,10]"
    # Calculate the number of bins needed to cover the data range
    data_range: int = data_delay.max() - data_delay.min()
    bin_count: int = int(np.ceil(data_range / bin_size)) or 1
    # Create a histogram with a specified bin size
    counts, bin_edges = np.histogram(
        data_delay,
        bins=bin_count,
        range=(data_delay.min(), data_delay.max()),
    )
    threshold: float = len(data_delay) * percentage_threshold / 100
    high_density_bins: NDArray[np.int_] = np.where(counts > threshold)[0]
    # Find consecutive groups of high-density bins and group them together
    groups = np.split(
        high_density_bins,
        np.where(np.diff(high_density_bins) != 1)[0] + 1,
    )
    # Filter out the data in each detected group
    masks_of_horizontal_lines: list[NDArray[np.bool_]] = []
    for group in groups:
        if group.size == 0:
            continue
        low_bound: NDArray[np.bool_] = data_delay >= bin_edges[group[0]]
        high_bound: NDArray[np.bool_] = data_delay <= bin_edges[group[-1] + 1]
        mask: NDArray[np.bool_] = low_bound & high_bound
        masks_of_horizontal_lines.append(mask)
    return tuple(masks_of_horizontal_lines)


@validate_inputs
def recursive_sigma_filter(
    data: NDArray,
    *,
    sigma_mult: float,
    max_iter: int = 100,
) -> tuple[NDArray, int]:
    """
    Recursively filter out values outside the ±n_sigma range of gaussian fit until convergence.
    :param data: Data to process, in the FW standard format.
    :param sigma_mult: Standard deviation multiplier that defines the range of values to keep (keyword-only).
    :param max_iter: Maximum number of iterations to prevent infinite loops (keyword-only).
    :return: Filtered data and number of filtering iterations.
    """
    assert max_iter > 0, "Max iterations must be positive"
    new_data: NDArray = data.copy()
    prev_data_len: int = 0
    iteration: int = 0
    # Iterate until the data is no longer changing in size
    while prev_data_len != len(new_data):
        prev_data_len = len(new_data)
        values: NDArray = new_data["femto"]
        mn: float = float(np.mean(values))
        std: float = float(np.std(values))
        # Filter out the outliers from the new_data
        low_filter: NDArray[np.bool_] = np.asarray(values >= mn - sigma_mult * std)
        high_filter: NDArray[np.bool_] = np.asarray(values <= mn + sigma_mult * std)
        new_data = new_data[low_filter & high_filter]
        iteration += 1
        if iteration == max_iter:
            raise RuntimeError(f"Max iterations reached!: {iteration}")
    check_data_structure(new_data)
    return new_data, iteration


@validate_inputs
def remove_drift(data: NDArray, deg: int = 1) -> NDArray:
    """
    Remove a polynomial drift/trend from a time series by least-squares fitting a
    polynomial to the data and subtracting it, leaving only the residuals.
    :param data: Data to be processed, in the FW standard format.
    :param deg: Degree of the polynomial fit used to model the drift.
    :return: Values with the fitted polynomial drift removed (residuals), the same length as input.
    """
    assert data.shape[0] > deg, f"Required >={deg + 1} points"
    # Center seconds around the first sample, so polyfit isn't fitting large absolute
    # timestamps, which can lose precision; centering doesn't change the residuals.
    seconds = data["seconds"] - data["seconds"][0]
    coefficients = np.polyfit(seconds, data["femto"], deg=deg)
    new_data = data.copy()
    residual = data["femto"] - np.polyval(coefficients, seconds)
    new_data["femto"] = np.round(residual)
    check_data_structure(new_data)
    return new_data
