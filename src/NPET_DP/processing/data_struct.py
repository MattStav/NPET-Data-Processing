from numpy.lib.recfunctions import unstructured_to_structured
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator, PrivateAttr

from NPET_DP.processing.calculations import (
    discard_rows_until_first_col_match,
    calculate_delay,
    detect_signal,
    recursive_sigma_filter,
    process_overflow,
    is_continuous,
    remove_drift,
)
from NPET_DP.processing.helpers import (
    import_data,
    DATA_TYPE,
    check_data_structure,
    _UNITS_TYPE,
    auto_scale_data,
    get_unit,
    auto_scale_num,
)


class NPETData(BaseModel):
    """Data structure for NPET data."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    seconds: NDArray[np.int_]
    femto: NDArray[np.int_]

    @model_validator(mode="after")
    def _lock_arrays(self) -> "NPETData":
        self.seconds.flags.writeable = False
        self.femto.flags.writeable = False
        return self

    def __repr__(self) -> str:
        return f"NPETData(len={len(self.structured_arr)})"

    def __len__(self) -> int:
        sec_len: int = len(self.seconds)
        femto_len: int = len(self.femto)
        assert sec_len == femto_len, f"Invalid data len: {sec_len} != {femto_len}"
        return sec_len

    @property
    def sc_femto(self) -> tuple[NDArray[np.floating], _UNITS_TYPE]:
        """Scaled femtoseconds delay values and their units."""
        scaled, sc_iter = auto_scale_data(self.femto)
        return scaled, get_unit("fs", sc_iter)

    @property
    def sc_mean(self) -> tuple[float, _UNITS_TYPE]:
        """Mean of the femtosecond delay values and its units."""
        mean_value = np.mean(self.femto)
        scaled_mean, sc_iter = auto_scale_num(mean_value)
        return float(scaled_mean), get_unit("fs", sc_iter)

    @property
    def sc_std(self) -> tuple[float, _UNITS_TYPE]:
        """Standard deviation of the femtosecond delay values and its units."""
        std_value = np.std(self.femto)
        scaled_std, sc_iter = auto_scale_num(std_value)
        return float(scaled_std), get_unit("fs", sc_iter)

    @property
    def structured_arr(self) -> NDArray:
        """Structured array holding the measured seconds and femtoseconds delay values in two columns."""
        arr: NDArray = np.column_stack((self.seconds, self.femto))
        struct_arr: NDArray = unstructured_to_structured(arr, dtype=np.dtype(DATA_TYPE))
        check_data_structure(struct_arr)
        return struct_arr

    @property
    def sc_structured_arr(self) -> tuple[NDArray, _UNITS_TYPE]:
        """Structured array holding the scaled seconds and femtoseconds delay values in two columns."""
        sc_femto, unit = self.sc_femto
        arr: NDArray = np.column_stack((self.seconds, sc_femto))
        struct_arr: NDArray = unstructured_to_structured(arr, dtype=np.dtype(DATA_TYPE))
        check_data_structure(struct_arr)
        return struct_arr, unit

    @classmethod
    def from_path(cls, data_path: Path, seconds_add: int = 0) -> "NPETData":
        """
        Import NPET data from a file.
        :param data_path: Path to the data file
        :param seconds_add: Number of seconds to add to each epoch, can be positive or negative
        :return: NPETData object
        """
        data: NDArray = import_data(data_path, seconds_add)
        return cls.from_structured_arr(data)

    @classmethod
    def from_structured_arr(cls, structured_arr: NDArray) -> "NPETData":
        """
        Init NPETData from a structured array.
        :param structured_arr: Structured array holding the measured seconds and femtoseconds delay values in two columns.
        :return: NPETData object
        """
        check_data_structure(structured_arr)
        return cls(seconds=structured_arr["seconds"], femto=structured_arr["femto"])

    @classmethod
    def empty(cls) -> "NPETData":
        """Create an empty NPETData object."""
        return cls(
            seconds=np.array([], dtype=np.int_), femto=np.array([], dtype=np.int_)
        )

    def discard_rows_until_ref_match(
        self,
        data_ref: "NPETData",
    ) -> tuple["NPETData", int]:
        """
        Discard rows from the data until the first row of the first column matches the reference data.
        :param data_ref: Reference data, the first row of the first column is used to match.
        """
        ret, discarded = discard_rows_until_first_col_match(
            data_to_process=self.structured_arr,
            data_ref=data_ref.structured_arr,
        )
        return NPETData.from_structured_arr(ret), discarded

    def calc_delay_start(self, *, stop: "NPETData", frequency: int) -> "NPETData":
        """
        Calculate the delay between this data (start) and the given stop data.
        :param stop: Stop data to calculate the delay against
        :param frequency: Frequency of the data
        :return: NPETData of the calculated delays
        """
        ret = calculate_delay(
            data_start=self.structured_arr,
            data_stop=stop.structured_arr,
            frequency=frequency,
        )
        return NPETData.from_structured_arr(ret)

    def calc_delay_stop(self, *, start: "NPETData", frequency: int) -> "NPETData":
        """
        Calculate the delay between the given start data and this data (stop).
        :param start: Start data to calculate the delay against
        :param frequency: Frequency of the data
        :return: NPETData of the calculated delays
        """
        ret = calculate_delay(
            data_start=start.structured_arr,
            data_stop=self.structured_arr,
            frequency=frequency,
        )
        return NPETData.from_structured_arr(ret)

    def detect_signal(
        self,
        bin_size: int = 40_000,  # fs
        percentage_threshold: float = 0.15,
    ) -> tuple[NDArray[np.bool_], ...]:
        """
        Detect signals in the delay data. Delays where data counts are above the threshold are considered signals.
        :param bin_size: The size of the bins in femtoseconds into which the data will be split.
        :param percentage_threshold: The percentage of data that must be in a bin to be considered a signal.
        :return: Boolean masks indicating detected signals.
        """
        return detect_signal(
            self.femto,
            bin_size=bin_size,
            percentage_threshold=percentage_threshold,
        )

    def define_signal_range(
        self,
        signal_mask: NDArray[np.bool_],
    ) -> tuple[float, float]:
        """
        Calculate a range of data around the detected signal.
        :param signal_mask: Boolean mask indicating the detected signal
        :return: A tuple defining the range min and max values
        """
        signal_values: NDArray[np.int_] = self.femto[signal_mask]
        range_center: float = (signal_values.max() + signal_values.min()) / 2
        new_range_size: float = (signal_values.max() - signal_values.min()) * 8
        return (
            range_center - new_range_size * 2 / 5,
            range_center + new_range_size * 3 / 5,
        )

    def filter_range(self, filter_mask: NDArray[np.bool_]) -> "NPETData":
        """
        Filter the data based on the given mask.
        :param filter_mask: Boolean mask indicating which data points to keep
        :return: NPETData object with filtered data
        """
        filtered_seconds = self.seconds[filter_mask]
        filtered_femto = self.femto[filter_mask]
        return NPETData(seconds=filtered_seconds, femto=filtered_femto)

    def recursive_sigma_filter(
        self,
        sigma_mult: float,
        max_iter: int = 100,
    ) -> tuple["NPETData", int]:
        """
        Perform recursive sigma filtering on the data.
        :param sigma_mult: The sigma multiplier for the filtering
        :param max_iter: The maximum number of iterations
        :return: NPETData object with filtered data and the number of iterations
        """
        res, sig_iter = recursive_sigma_filter(
            self.structured_arr,
            sigma_mult=sigma_mult,
            max_iter=max_iter,
        )
        return NPETData.from_structured_arr(res), sig_iter

    def process_incremental_overflow(self) -> "NPETData":
        """Process the data to handle incremental overflows."""
        res = process_overflow(self.structured_arr)
        return NPETData.from_structured_arr(res)

    def is_seconds_continuous(self, expected_diff: int = 1) -> bool:
        """Return True if the `seconds` data is continuous, False otherwise."""
        return is_continuous(self.seconds, expected_diff=expected_diff)

    def compensate_drift(self, pol_deg: int = 1) -> "NPETData":
        """
        Remove drift from the data using polynomial regression.
        :param pol_deg: Degree of the polynomial used for drift removal.
        :return: NPETData object with drift removed.
        """
        no_drift = remove_drift(self.structured_arr, deg=pol_deg)
        return NPETData.from_structured_arr(no_drift)

    def femto_not_in(self, other: "NPETData") -> "NPETData":
        """
        Return a new NPETData object containing only the rows that are not present in the other NPETData object.
        :param other: The other NPETData object to compare against.
        :return: NPETData object with rows not present in the other object.
        """
        mask = ~np.isin(self.femto, other.femto)
        filtered_seconds = self.seconds[mask]
        filtered_femto = self.femto[mask]
        return NPETData(seconds=filtered_seconds, femto=filtered_femto)
