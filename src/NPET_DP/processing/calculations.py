import math

import numpy as np
from numpy.typing import NDArray

from NPET_DP.framework.constants import FEMTO
from NPET_DP.processing.helpers import (
    DATA_TYPE,
    check_data_structure,
    validate_inputs,
)


@validate_inputs
def process_overflow(data: NDArray) -> NDArray:
    """Process the data overflowing into the previous second.

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
    """Check whether one column data is continuous, i.e., all values are consecutive.

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
    """Discard rows until the first column of the data matches the first column of the reference data.

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
    """Calculate the delay between two sets of EPOCH measurement data.

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
    max_stop_scan: int = frequency * 2 + 10
    half_femto: float = FEMTO / 2

    # Field access on a numpy structured scalar (epoch_start["seconds"]) is much
    # slower than plain int indexing, and this loop is O(N * frequency) in the
    # worst case, so work on plain Python lists/ints instead.
    start_seconds: list[int] = data_start["seconds"].tolist()
    start_femto: list[int] = data_start["femto"].tolist()
    stop_seconds: list[int] = data_stop["seconds"].tolist()
    stop_femto: list[int] = data_stop["femto"].tolist()
    n_stop: int = len(stop_seconds)

    results: list[tuple[int, int]] = []
    stop_idx: int = 0
    for s_sec, s_femto in zip(start_seconds, start_femto, strict=True):
        # If no match was found in one measurement interval plus some margin,
        # proceed to the next start epoch. The current start epoch is discarded.
        limit: int = min(stop_idx + max_stop_scan, n_stop)
        for i in range(stop_idx, limit):
            e_sec = stop_seconds[i]
            e_femto = stop_femto[i]
            # Case when the measurements happened on the same second
            if e_sec == s_sec:
                diff: int = e_femto - s_femto
                diff_abs: int = abs(diff)
            # Case when measurements happened on a different second, but they fulfill conditions for potential cross
            elif (
                e_sec > s_sec
                and e_femto > s_femto
                and (e_femto > half_femto or s_femto > half_femto)
            ):
                diff_abs = FEMTO - abs(e_femto - s_femto)
                # Choose the sign of the difference based on which epoch is earlier
                diff = -diff_abs if e_sec < s_sec else diff_abs
            # Case when there is no chance of matching the epochs
            else:
                continue
            # If the difference is bigger than the possible detection interval,
            # skip this combination and proceed to the next stop epoch
            if diff_abs > searched_interval_size:
                continue
            results.append((s_sec, diff))
            # Don't process the same epochs again, only consider the data after the current epoch.
            # The measurements can't travel back in time :)
            stop_idx = i + 1
            # Break to proceed to the next measurement set
            break
    diff_mat: NDArray = np.array(results, dtype=DATA_TYPE)
    check_data_structure(diff_mat)
    return diff_mat


def detect_signal(
    data_delay: NDArray[np.int_],
    *,
    bin_size: int = 1_000_000,  # fs = 1 ns
    init_thresh: float = 0.08,
    min_thresh: float = 0.5,
    max_thresh: float = 20,
    signal_upper_bound: int = 1_000_000,  # fs = 1 ns
) -> tuple[NDArray[np.bool_], ...]:
    """Detect signals in the delay data by identifying horizontal lines (clusters of similar delay values).

    Delays where data counts are above the threshold are considered signals.
    The default values are selected for detecting a 20 ps width pulse with at least 2% return rate.
    Any detected signal wider than max_signal_width is refined: the detection threshold is doubled
    and re-applied to just that signal's own data, which can split it into several narrower signals.
    Each resulting signal is checked again and refined further if it is still too wide, up until
    the percentage threshold would exceed 20%, at which point the signal is accepted as-is.
    Final signals containing less than 1.5% of the data are discarded.
    :param data_delay: Data to be processed, the femtoseconds delay column from the data.
    :param bin_size: The size of the bins in femtoseconds into which the data will be split (keyword-only).
    :param init_thresh: The percentage of data that must be in a bin to be considered a signal (keyword-only).
    :param max_thresh: The maximum percentage of data that can be in a bin to be considered a signal.
    The iteration stops once the required percentage crosses this threshold. (keyword-only)
    :param min_thresh: The minimum percentage of data that must be in a bin to be considered a signal (keyword-only).
    :param signal_upper_bound: Maximum signal width in femtoseconds before the threshold is doubled and the signal re-detected (keyword-only).
    :return: Boolean masks indicating detected signals.
    Each mask corresponds to a detected signal (horizontal line) in the data.
    """
    assert data_delay.ndim == 1, "Data must be 1D"
    assert bin_size > 0, "Bin size must be positive"
    assert 0 <= init_thresh <= 10, "Percentage threshold must be [0,10]"
    assert 0 <= min_thresh <= 10, "Percentage threshold must be [0,10]"
    assert 0 <= max_thresh <= 100, "Percentage threshold must be [0,100]"
    assert signal_upper_bound > 0, "Max signal width must be positive"

    def find_groups(mask: NDArray[np.bool_], thresh: float) -> list[NDArray[np.bool_]]:
        """Detect horizontal lines within the given subset of data_delay, above the given threshold.

        :param mask: Boolean mask indicating which data points to consider.
        :param thresh: The percentage of data that must be in a bin to be considered a signal.
        :return: List of boolean masks indicating detected signals within the subset.
        """
        subset: NDArray[np.int_] = data_delay[mask]
        # Calculate the number of bins needed to cover the data range
        data_range: int = subset.max() - subset.min()
        bin_count: int = int(np.ceil(data_range / bin_size)) or 1
        # Create a histogram with a specified bin size
        counts, bin_edges = np.histogram(
            subset,
            bins=bin_count,
            range=(subset.min(), subset.max()),
        )
        threshold: float = len(data_delay) * thresh / 100
        high_density_bins: NDArray[np.int_] = np.where(counts > threshold)[0]
        # Find consecutive groups of high-density bins and group them together
        groups = np.split(
            high_density_bins,
            np.where(np.diff(high_density_bins) != 1)[0] + 1,
        )
        # Filter out the data in each detected group
        found: list[NDArray[np.bool_]] = []
        for group in groups:
            if group.size == 0:
                continue
            low_bound: NDArray[np.bool_] = data_delay >= bin_edges[group[0]]
            high_bound: NDArray[np.bool_] = data_delay <= bin_edges[group[-1] + 1]
            found.append(mask & low_bound & high_bound)
        return found

    full_mask: NDArray[np.bool_] = np.ones(data_delay.shape, dtype=np.bool_)
    # Queue of (mask, percentage_threshold) pairs still needing to be checked/refined
    pending: list[tuple[NDArray[np.bool_], float]] = [
        (mask, init_thresh) for mask in find_groups(full_mask, init_thresh)
    ]
    masks_of_horizontal_lines: list[NDArray[np.bool_]] = []
    while pending:
        mask, percentage_threshold = pending.pop(0)
        signal: NDArray[np.int_] = data_delay[mask]
        width: int = signal.max() - signal.min()
        if width > signal_upper_bound and percentage_threshold * 2 <= max_thresh:
            # Signal too wide: double the threshold and re-detect within just this signal's data,
            # which may split it into several narrower signals to check again.
            percentage_threshold *= 2
            pending.extend(
                (sub_mask, percentage_threshold)
                for sub_mask in find_groups(mask, percentage_threshold)
            )
        else:
            masks_of_horizontal_lines.append(mask)
    # Discard final signals that end up too small to be meaningful
    min_final_threshold: float = len(data_delay) * min_thresh / 100
    return tuple(
        mask
        for mask in masks_of_horizontal_lines
        if np.sum(mask) >= min_final_threshold
    )


@validate_inputs
def recursive_sigma_filter(
    data: NDArray,
    *,
    sigma_mult: float,
    max_iter: int = 100,
) -> tuple[NDArray, int]:
    """Recursively filter out values outside the ±n_sigma range of gaussian fit until convergence.

    :param data: Data to process, in the FW standard format.
    :param sigma_mult: Standard deviation multiplier that defines the range of values to keep (keyword-only).
    :param max_iter: Maximum number of iterations to prevent infinite loops (keyword-only).
    :return: Filtered data and number of filtering iterations.
    """
    assert max_iter > 0, "Max iterations must be positive"
    assert data.size > 0, "Data must not be empty"
    # Work on a contiguous copy of just the relevant column (instead of
    # re-filtering the whole structured array every iteration) and only
    # materialize the filtered structured result once, at the end.
    values: NDArray[np.int64] = data["femto"]
    idx: NDArray[np.int64] = np.arange(values.size).astype(np.int64)
    prev_data_len: int = 0
    iteration: int = 0
    # Iterate until the data is no longer changing in size
    while prev_data_len != len(values):
        if iteration == max_iter:
            raise RuntimeError(f"Max iterations reached!: {iteration}")
        prev_data_len = len(values)
        mn: int = round(values.mean())
        diff: NDArray[np.int64] = values - mn
        std: float = np.sqrt(diff.dot(diff) / diff.size)
        # Filter out the outliers
        keep: NDArray[np.bool_] = np.abs(diff) <= sigma_mult * std
        values = values[keep]
        idx = idx[keep]
        iteration += 1
    new_data: NDArray = data[idx]
    check_data_structure(new_data)
    return new_data, iteration


@validate_inputs
def remove_drift(data: NDArray, deg: int = 1) -> NDArray:
    """Remove drift from data.

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


def interp_crossing(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    y_target: float,
) -> float:
    """Linearly interpolate the x-value where the line through (x1, y1)-(x2, y2) crosses y_target.

    :param x1: X-coordinate of the first point.
    :param x2: X-coordinate of the second point.
    :param y1: Y-coordinate of the first point.
    :param y2: Y-coordinate of the second point.
    :param y_target: Y-value to find the crossing x-value for.
    :return: X-value at which the line reaches y_target.
    """
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def get_bin_count(data_spread: float, target_bin_size_fs: int = 10_000) -> int:
    """Calculate the bin count.

    Calculate the number of bins for a histogram based on the data spread and target bin size.
    :param data_spread: The values spread across the data.
    :param target_bin_size_fs: Target bin size in femtoseconds.
    :return: Number of bins for the histogram.
    """
    # Difference between the max and min delay
    if data_spread > 50_000_000:  # fs
        # If there's too much data (>50 ns), then adjust the bin size to match 1000 bins
        bin_size: float = data_spread / 1000
    else:
        # Otherwise, use the supplied bin size in femtoseconds
        bin_size: float = target_bin_size_fs
    bin_count: int = math.floor(data_spread / bin_size)
    return bin_count


@validate_inputs
def calc_fwhm(data: NDArray) -> tuple[int, float]:
    """Calculate the full-width half-maximum (FWHM) of the data.

    :param data: Data to be processed, in the FW standard format.
    :return: The place and value of the FWHM in femtoseconds.
    """
    femto = data["femto"]
    bins = get_bin_count(femto.max() - femto.min())
    counts, bin_edges = np.histogram(femto, bins=bins)
    bin_centers: NDArray[np.floating] = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    half_max: float = counts.max() / 2
    above_half_max: NDArray[np.int_] = np.where(counts >= half_max)[0]
    left_idx, right_idx = above_half_max[0], above_half_max[-1]
    # If the peak's above-half-max region touches the outer edge, there is no
    # neighboring bin to interpolate against, so fall back to the bin edge itself.
    signal_min: float = (
        float(bin_edges[0])
        if left_idx == 0
        else interp_crossing(
            float(bin_centers[left_idx - 1]),
            float(bin_centers[left_idx]),
            float(counts[left_idx - 1]),
            float(counts[left_idx]),
            half_max,
        )
    )
    signal_max: float = (
        float(bin_edges[-1])
        if right_idx == len(counts) - 1
        else interp_crossing(
            float(bin_centers[right_idx]),
            float(bin_centers[right_idx + 1]),
            float(counts[right_idx]),
            float(counts[right_idx + 1]),
            half_max,
        )
    )
    fwhm: float = signal_max - signal_min
    fwhm_center: int = round(bin_centers[np.argmax(counts)])
    return fwhm_center, fwhm
