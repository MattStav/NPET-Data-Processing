import numpy as np

from NPET_DP.epoch_processing.helper_processing import process_overflow
from NPET_DP.epoch_processing.helper_funcs import (
    import_data,
    auto_scale_data,
    auto_scale_num,
    get_unit,
)
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "test_file, expected_mean, expected_mean_unit, expected_std, expected_std_unit",
    [
        ("test_PPS.out", -5.67741, "ps", 18.2123, "ps"),
        ("test_PPS_long.out", -36.52495, "ps", 26.8100, "ps"),
        ("test_PPS_nonzero.out", -498.09178, "ms", 24.1224, "ps"),
    ],
)
def test_pps_workflow(
    data_dir: Path,
    test_file: str,
    expected_mean: float,
    expected_mean_unit: str,
    expected_std: float,
    expected_std_unit: str,
):
    data_p: Path = data_dir / test_file
    assert data_p.exists()
    data_pps = import_data(data_p)
    processed = process_overflow(data_pps)
    sc_mean, sc_mean_iter = auto_scale_num(np.mean(processed["femto"]))
    assert np.isclose(sc_mean, expected_mean)
    mean_unit = get_unit("fs", sc_mean_iter)
    assert mean_unit == expected_mean_unit, (
        f"Expected mean unit to be {expected_mean_unit}, got {mean_unit} instead."
    )
    sc_std, sc_std_iter = auto_scale_num(np.std(processed["femto"]))
    assert np.isclose(sc_std, expected_std)
    std_unit = get_unit("fs", sc_std_iter)
    assert std_unit == expected_std_unit, (
        f"Expected std unit to be {expected_std_unit}, got {std_unit} instead."
    )
