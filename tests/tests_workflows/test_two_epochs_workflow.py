from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from NPET_DP.processing.data_struct import NPETData
from NPET_DP.processing.helpers import import_data, auto_scale_num, get_unit

from NPET_DP.processing.calculations import (
    discard_rows_until_first_col_match,
    calculate_delay,
    detect_signal,
    recursive_sigma_filter,
)


def test_two_epochs(data_dir):
    start_p: Path = data_dir / "test_data_START.out"
    stop_p: Path = data_dir / "test_data_STOP.out"
    data_start = import_data(start_p)
    data_stop = import_data(stop_p)
    data_stop, discarded = discard_rows_until_first_col_match(
        data_ref=data_start,
        data_to_process=data_stop,
    )
    assert discarded == 7729, f"Unexpected number of discarded points: {discarded}"
    delays = calculate_delay(
        data_start=data_start,
        data_stop=data_stop,
        frequency=500,
    )
    assert len(delays) == 32220, f"Unexpected number of points: {len(delays)}"
    autodetection: tuple[NDArray[np.bool_], ...] = detect_signal(delays["femto"])
    assert len(autodetection) == 1, f"Unexpected auto-detections: {len(autodetection)}"
    signal_values: NDArray[np.int_] = delays["femto"][autodetection[0]]
    range_center: float = (signal_values.max() + signal_values.min()) / 2
    new_range_size: float = (signal_values.max() - signal_values.min()) * 8
    min_delay = range_center - new_range_size * 2 / 5
    max_delay = range_center + new_range_size * 3 / 5
    assert np.isclose(min_delay, 58824679.9), f"Unexpected min_delay: {min_delay}"
    assert np.isclose(max_delay, 60729103.9), f"Unexpected max_delay: {max_delay}"
    mask = (min_delay <= delays["femto"]) & (delays["femto"] <= max_delay)
    selected_delays = delays[mask]
    delay_sel = len(selected_delays)
    assert delay_sel == 2419, f"Unexpected delay sel.: {delay_sel}"
    sigma_filtered, sigma_iter = recursive_sigma_filter(selected_delays, sigma_mult=2.2)
    sigma_f_num = len(sigma_filtered)
    assert sigma_f_num == 1567, f"Unexpected sigma filtered: {sigma_f_num}"
    assert sigma_iter == 18, f"Unexpected sigma iterations: {sigma_iter}"
    mean = np.mean(sigma_filtered["femto"])
    sc_mean, mean_iter = auto_scale_num(mean)
    mean_unit = get_unit("fs", mean_iter)
    assert np.isclose(sc_mean, 59.5264), f"Unexpected mean: {mean}"
    assert mean_unit == "ns", f"Unexpected mean unit: {mean_unit}"
    std = np.std(sigma_filtered["femto"])
    sc_std, std_iter = auto_scale_num(std)
    std_unit = get_unit("fs", std_iter)
    assert np.isclose(sc_std, 17.1517), f"Unexpected std: {sc_std}"
    assert std_unit == "ps", f"Unexpected std unit: {std_unit}"
    ret_rate: float = 100 * sigma_f_num / len(delays)
    assert np.isclose(ret_rate, 4.86, atol=0.005), f"Unexpected return rate: {ret_rate}"


def test_two_epochs_npet_data(data_dir: Path):
    start_p: Path = data_dir / "test_data_START.out"
    stop_p: Path = data_dir / "test_data_STOP.out"
    data_start = NPETData.from_path(start_p)
    data_stop = NPETData.from_path(stop_p)
    data_stop, discarded = data_stop.discard_rows_until_ref_match(data_start)
    assert discarded == 7729, f"Unexpected number of discarded points: {discarded}"
    delays = data_stop.calc_delay_stop(start=data_start, frequency=500)
    assert len(delays) == 32220, f"Unexpected number of points: {len(delays)}"
    autodetection: tuple[NDArray[np.bool_], ...] = delays.detect_signal()
    assert len(autodetection) == 1, f"Unexpected auto-detections: {len(autodetection)}"
    signal_range = delays.define_signal_range(autodetection[0])
    assert np.isclose(signal_range[0], 58824679.9), (
        f"Unexpected min_delay: {signal_range[0]}"
    )
    assert np.isclose(signal_range[1], 60729103.9), (
        f"Unexpected max_delay: {signal_range[1]}"
    )
    mask = (signal_range[0] <= delays.femto) & (delays.femto <= signal_range[1])
    selected_delays = delays.filter_range(mask)
    delay_sel = len(selected_delays)
    assert delay_sel == 2419, f"Unexpected delay sel.: {delay_sel}"
    sigma_filtered, sigma_iter = selected_delays.recursive_sigma_filter(2.2)
    sigma_f_num = len(sigma_filtered)
    assert sigma_f_num == 1567, f"Unexpected sigma filtered: {sigma_f_num}"
    assert sigma_iter == 18, f"Unexpected sigma iterations: {sigma_iter}"
    sc_mean, mean_unit = sigma_filtered.sc_mean
    assert np.isclose(sc_mean, 59.5264), f"Unexpected mean: {sc_mean}"
    assert mean_unit == "ns", f"Unexpected mean unit: {mean_unit}"
    sc_std, std_unit = sigma_filtered.sc_std
    assert np.isclose(sc_std, 17.1517), f"Unexpected std: {sc_std}"
    assert std_unit == "ps", f"Unexpected std unit: {std_unit}"
    ret_rate: float = 100 * sigma_f_num / len(delays)
    assert np.isclose(ret_rate, 4.86, atol=0.005), f"Unexpected return rate: {ret_rate}"
