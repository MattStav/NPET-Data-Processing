import numpy as np
import numpy.typing as npt
import pytest

from NPET_DP.processing.helpers import DATA_TYPE
from NPET_DP.processing.calculations import (
    calculate_delay,
    detect_signal,
    discard_rows_until_first_col_match,
    is_continuous,
    process_overflow,
    recursive_sigma_filter,
    remove_drift,
)


def _structured_data(values: list[int]) -> npt.NDArray:
    """Build a DATA_TYPE structured array with the given femto values."""
    return np.array(list(zip(range(len(values)), values, strict=True)), dtype=DATA_TYPE)


def test_process_overflow():
    """Test that the function correctly processes overflows in the data."""
    data = np.array(
        [
            (0, 999999999999900),
            (2, 100),
            (3, 200),
            (3, 999999999999700),
            (5, 100),
            (6, 50),
            (6, 999999999999500),
            (8, 500),
        ],
        dtype=[("seconds", np.int64), ("femto", np.int64)],
    )
    assert np.array_equal(
        process_overflow(data),
        np.array(
            [
                (1, -100),
                (2, 100),
                (3, 200),
                (4, -300),
                (5, 100),
                (6, 50),
                (7, -500),
                (8, 500),
            ],
            dtype=DATA_TYPE,
        ),
    ), "Overflow processing failed"


@pytest.mark.parametrize(
    "data, expected_result",
    ((np.array([1, 2, 3, 4]), True), (np.array([1, 2, 4, 5]), False)),
)
def test_is_continuous(data, expected_result):
    """Test array continuity check."""
    assert is_continuous(data) == expected_result


@pytest.mark.parametrize(
    "ref_rows,mismatch_rows,expected_rows,expected_discarded",
    [
        pytest.param(
            [(10, 100), (11, 101), (12, 102)],
            [(7, 70), (8, 80), (10, 100), (11, 101)],
            [(10, 100), (11, 101)],
            2,
            id="discard-two-leading-rows",
        ),
        pytest.param(
            [(5, 50), (6, 60)],
            [(5, 500), (6, 600), (7, 700)],
            [(5, 500), (6, 600), (7, 700)],
            0,
            id="already-matching-at-first-row",
        ),
        pytest.param(
            [(12, 120), (13, 130)],
            [(1, 10), (2, 20), (3, 30), (12, 120)],
            [(12, 120)],
            3,
            id="discard-until-late-match",
        ),
        pytest.param(
            [(100, 1)],
            [(20, 2), (40, 3), (60, 4), (100, 5), (120, 6)],
            [(100, 5), (120, 6)],
            3,
            id="single-row-reference",
        ),
    ],
)
def test_discard_rows_until_first_col_match(
    ref_rows: list[tuple[int, int]],
    mismatch_rows: list[tuple[int, int]],
    expected_rows: list[tuple[int, int]],
    expected_discarded: int,
):
    """Verifies that the function correctly discards rows until the first column matches."""
    ref_data = np.array(ref_rows, dtype=DATA_TYPE)
    mismatch_data = np.array(mismatch_rows, dtype=DATA_TYPE)
    filtered, discarded = discard_rows_until_first_col_match(
        data_ref=ref_data,
        data_to_process=mismatch_data,
    )
    assert discarded == expected_discarded
    assert np.array_equal(filtered, np.array(expected_rows, dtype=DATA_TYPE))


def test_discard_rows_until_first_col_match_returns_full_tail_from_match_point():
    """Verifies that the function keeps the matched row and everything after it."""
    ref_data = np.array([(8, 0), (9, 0)], dtype=DATA_TYPE)
    mismatch_data = np.array(
        [(5, 50), (6, 60), (8, 80), (9, 90), (10, 100)], dtype=DATA_TYPE
    )
    filtered, discarded = discard_rows_until_first_col_match(
        data_ref=ref_data,
        data_to_process=mismatch_data,
    )
    assert discarded == 2
    assert np.array_equal(
        filtered,
        np.array([(8, 80), (9, 90), (10, 100)], dtype=DATA_TYPE),
    )


def test_discard_rows_until_first_col_match_raises_index_error_when_no_match():
    """The function should raise IndexError if the input data never reaches the reference value."""
    ref_data = np.array([(10, 0)], dtype=DATA_TYPE)
    mismatch_data = np.array([(1, 10), (2, 20), (3, 30)], dtype=DATA_TYPE)
    with pytest.raises(IndexError):
        discard_rows_until_first_col_match(
            data_ref=ref_data,
            data_to_process=mismatch_data,
        )


def test_calculate_delay():
    """Test calculate_delay with simple matching data"""
    val: int = 100_000_000_000_000
    data_start = np.array([(i, val) for i in range(12)], dtype=DATA_TYPE)
    # Stop data with a constant 1000 femto delay
    data_stop = np.array([(i, val + 1000) for i in range(12)], dtype=DATA_TYPE)
    delays: npt.NDArray = calculate_delay(
        data_start=data_start,
        data_stop=data_stop,
        frequency=10,
    )
    assert len(delays) == 12
    assert np.all(delays["femto"] == 1000)


def test_calculate_delay_wraparound():
    """Test calculate_delay with values wrapping around the second boundary (10+ points)."""
    val_start = 10
    val_stop = 999_999_999_999_990  # FEMTO - 10
    data_start = np.array([(i, val_start) for i in range(10)], dtype=DATA_TYPE)
    data_stop = np.array([(i + 1, val_stop) for i in range(10)], dtype=DATA_TYPE)
    delays: npt.NDArray = calculate_delay(
        data_start=data_start,
        data_stop=data_stop,
        frequency=10,
    )
    assert len(delays) == 10
    assert np.all(delays["femto"] == 20)


def test_calculate_delay_missing_and_out_of_range():
    """Test calculate_delay with missing points and out-of-range delays (10+ points)."""
    data_start = np.array(
        [(i, 500_000_000_000_000) for i in range(15)], dtype=DATA_TYPE
    )
    # Stop data:
    # 0-4: matched
    # 5: missing in stop
    # 6: matched
    # 7: out-of-range delay in stop (> 50e12)
    # 8-14: matched
    stop_list = []
    for i in range(5):
        stop_list.append((i, 500_000_000_000_100))  # delay 100
    # i=5 is skipped in stop
    stop_list.append((6, 499_999_999_999_900))  # i=6 matched
    stop_list.append((7, 560_000_000_000_000))  # i=7 out of range (>50e12)
    for i in range(8, 15):
        stop_list.append((i, 500_000_000_600_100))
    data_stop = np.array(stop_list, dtype=DATA_TYPE)
    delays: npt.NDArray = calculate_delay(
        data_start=data_start,
        data_stop=data_stop,
        frequency=10,
    )
    # Expected:
    # start 0-4: match (5 points)
    # start 5: no match (stop 5 is missing, stop 6 is at sec 6, mismatch)
    # start 6: match with stop 6 (1 point)
    # start 7: no match (stop 7 is out of range)
    # start 8-14: match (7 points)
    # Total expected: 5 + 1 + 7 = 13 points
    assert len(delays) == 13, "There should be 13/15 points"
    assert np.all(delays["femto"][0:4] == 100)
    assert delays["femto"][5] == -100
    assert np.all(delays["femto"][6:13] == 600_100)


def test_detect_signal_single_cluster():
    """Test detect_signal with a single clear cluster of delays."""
    # 100 points, 20 at 1000, 80 at 500000
    data: npt.NDArray[np.int_] = np.array([1000] * 20 + [500000] * 80, dtype=np.int_)
    # Any bin with >= 1 point will be selected if we use percentage_threshold=0.15
    masks = detect_signal(data, bin_size=40000, percentage_threshold=0.15)
    assert len(masks) == 2
    assert np.sum(masks[0]) == 20
    assert np.sum(masks[1]) == 80


def test_detect_signal_no_signal():
    """Test detect_signal when no bin exceeds the threshold."""
    # 100 points spread out
    data: npt.NDArray[np.int_] = np.arange(0, 1000000, 10000, dtype=np.int_).astype(
        np.int_
    )
    masks = detect_signal(data, bin_size=5000, percentage_threshold=1.1)
    assert len(masks) == 0


def test_detect_signal_multiple_clusters():
    """Test detect_signal with multiple clusters."""
    data: npt.NDArray[np.int_] = np.concatenate(
        [np.full(30, 100000), np.full(30, 500000), np.full(30, 900000)]
    ).astype(np.int_)
    # 90 points total. threshold = 90 * 0.1 / 100 = 0.09.
    masks = detect_signal(data, bin_size=10000, percentage_threshold=0.1)
    assert len(masks) == 3
    for mask in masks:
        assert np.sum(mask) == 30


def test_detect_signal_consecutive_bins():
    """Test that consecutive high-density bins are grouped together."""
    data: npt.NDArray[np.int_] = np.array([5000] * 10 + [15000] * 10, dtype=np.int_)
    # total 20 points. threshold = 20 * 0.05 / 100 = 0.01.
    masks = detect_signal(data, bin_size=10000, percentage_threshold=0.05)
    assert len(masks) == 1
    assert np.sum(masks[0]) == 20


def test_detect_signal_overlapping_clusters():
    """Test detect_signal with clusters that are close but in distinct bins."""
    # bin_size = 10000.
    # Bins: [0, 10000), [10000, 20000), [20000, 30000)
    data: npt.NDArray[np.int_] = np.array([4000] * 10 + [25000] * 10, dtype=np.int_)
    # threshold = 20 * 0.05 / 100 = 0.01.
    # Bins 0 and 2 are selected. Bin 1 is NOT.
    # So they should NOT be grouped.
    masks = detect_signal(data, bin_size=10000, percentage_threshold=0.05)
    assert len(masks) == 2
    assert np.sum(masks[0]) == 10
    assert np.sum(masks[1]) == 10


def test_detect_signal_all_same_values():
    """Test detect_signal when all values are identical."""
    data: npt.NDArray[np.int_] = np.array([1000] * 100, dtype=np.int_)
    masks = detect_signal(data, bin_size=10000, percentage_threshold=0.1)
    assert len(masks) == 1
    assert np.sum(masks[0]) == 100


def test_recursive_sigma_filter_no_outliers():
    """Test recursive_sigma_filter when all data is within the sigma range."""
    data: npt.NDArray = _structured_data([100, 101, 99, 100, 100])
    # mean=100, std ~0.63. With sigma=2.2, range is ~[98.6, 101.4]. All are in.
    filtered, iterations = recursive_sigma_filter(data, sigma_mult=2.2)
    assert np.array_equal(filtered, data)
    assert iterations == 1


def test_recursive_sigma_filter_with_outliers():
    """Test recursive_sigma_filter with clear outliers."""
    # Data with a clear outlier: 1000
    data: npt.NDArray = _structured_data([100, 101, 99, 100, 100, 1000])
    # First pass: mean=250, std ~335. Range [250 - 2.2*335, 250 + 2.2*335] -> [-487, 987]
    # 1000 is filtered out.
    # Second pass: [100, 101, 99, 100, 100], mean=100, std=0.63. Range [98.6, 101.4]. All in.
    # Converged.
    filtered, iterations = recursive_sigma_filter(data, sigma_mult=2.2)
    assert np.array_equal(filtered, data[:5])
    assert iterations == 2


def test_recursive_sigma_filter_all_identical():
    """Test recursive_sigma_filter when all values are identical (std=0)."""
    data: npt.NDArray = _structured_data([100, 100, 100])
    filtered, iterations = recursive_sigma_filter(data, sigma_mult=2.2)
    assert np.array_equal(filtered, data)
    assert iterations == 1


def test_recursive_sigma_filter_max_iter():
    """Test that recursive_sigma_filter raises RuntimeError when max_iter is reached."""
    # We need a case that doesn't converge or just force it with max_iter=1
    data: npt.NDArray = _structured_data([100, 100, 100, 1000])
    # It would normally take 2 iterations. Setting max_iter=1 should trigger an error.
    with pytest.raises(RuntimeError, match="Max iterations reached!"):
        recursive_sigma_filter(data, sigma_mult=2.2, max_iter=1)


def test_recursive_sigma_filter_empty():
    """Test recursive_sigma_filter with empty data."""
    data: npt.NDArray = np.array([], dtype=DATA_TYPE)
    filtered, iterations = recursive_sigma_filter(data, sigma_mult=2.2)
    assert len(filtered) == 0
    assert iterations == 0


def test_recursive_sigma_filter_large_dataset():
    """Test recursive_sigma_filter with a large dataset (e.g., 100,000 points)."""
    # Create 100,000 points from a normal distribution + 1000 outliers
    np.random.seed(42)
    clean_data = np.random.normal(loc=1000, scale=10, size=100000).astype(np.int64)
    outliers = np.random.uniform(low=2000, high=3000, size=1000).astype(np.int64)
    values = np.concatenate([clean_data, outliers])
    np.random.shuffle(values)
    data = np.array(list(zip(range(len(values)), values, strict=True)), dtype=DATA_TYPE)
    filtered, iterations = recursive_sigma_filter(data, sigma_mult=3.0)
    # With sigma=3.0, we expect most of clean_data to stay and all outliers to be removed.
    # Outliers (2000-3000) are far from mean (1000) with std (10).
    assert len(filtered) < len(data)
    assert len(filtered) >= 99000  # Most clean data should remain
    assert np.all(filtered["femto"] < 1500)  # Outliers should be gone
    assert iterations > 1


def test_remove_drift_linear_trend():
    """A pure linear trend should be fully removed, leaving ~0 residuals."""
    time = np.arange(20)
    values = 3 * time + 5
    data = np.array(list(zip(time, values, strict=True)), dtype=DATA_TYPE)
    residuals = remove_drift(data)["femto"]
    assert np.allclose(residuals, 0, atol=1e-8)


def test_remove_drift_quadratic_trend():
    """A quadratic trend should be fully removed by the default degree-2 fit."""
    time = np.arange(20)
    values = time**2 - 3 * time + 5
    data = np.array(list(zip(time, values, strict=True)), dtype=DATA_TYPE)
    residuals = remove_drift(data, 2)["femto"]
    assert np.allclose(residuals, 0, atol=1e-6)


def test_remove_drift_recovers_added_noise():
    """After removing a linear trend, residuals should be noise-scale, not trend-scale."""
    rng = np.random.default_rng(42)
    time = np.arange(50)
    trend = 3 * time + 1  # grows up to ~148
    noise = rng.integers(-2, 3, size=time.shape)  # small integer noise
    values = trend + noise
    data = np.array(list(zip(time, values, strict=True)), dtype=DATA_TYPE)
    residuals = remove_drift(data, 1)["femto"]
    # A least-squares fit doesn't exactly recover the injected noise (it absorbs
    # part of it into the fitted coefficients), so only check the residual stays
    # within a small multiple of the noise scale rather than matching it exactly.
    assert np.max(np.abs(residuals)) < 5


def test_remove_drift_constant_data():
    """Data with no trend should return residuals of ~0."""
    time = np.arange(10)
    values = np.full(10, 7)
    data = np.array(list(zip(time, values, strict=True)), dtype=DATA_TYPE)
    residuals = remove_drift(data)["femto"]
    assert np.allclose(residuals, 0, atol=1e-8)


def test_remove_drift_custom_degree_leaves_unmodeled_component():
    """Fitting a lower degree than the data's actual trend should leave a residual curve."""
    time = np.arange(20)
    values = time**2
    data = np.array(list(zip(time, values, strict=True)), dtype=DATA_TYPE)
    residuals = remove_drift(data, deg=1)["femto"]
    assert not np.allclose(residuals, 0, atol=1e-3)


def test_remove_drift_invalid_shape_raises():
    """A non-structured array should raise a ValueError."""
    data = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        remove_drift(data)


def test_remove_drift_too_few_points_raises():
    """Fewer points than deg + 1 should raise an AssertionError."""
    data = np.array([(0, 1), (1, 2)], dtype=DATA_TYPE)
    with pytest.raises(AssertionError):
        remove_drift(data, deg=2)
