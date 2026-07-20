from numpy.lib.recfunctions import unstructured_to_structured
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from NPET_DP.processing.calculations import (
    discard_rows_until_first_col_match,
    calculate_delay,
    detect_signal,
    recursive_sigma_filter,
)
from NPET_DP.processing.helpers import import_data, DATA_TYPE, check_data_structure


class NPETData(BaseModel):
    """Data structure for NPET data."""

    __seconds: NDArray[np.int_]
    __femto: NDArray[np.int_]

    def __repr__(self) -> str:
        return f"NPETData(len={len(self.structured_arr)})"

    @property
    def seconds(self) -> NDArray[np.int_]:
        """1D array holding the measured seconds delay values in one column."""
        return self.__seconds

    @property
    def femto(self) -> NDArray[np.int_]:
        """1D array holding the measured femtoseconds delay values in one column."""
        return self.__femto

    @property
    def structured_arr(self) -> NDArray:
        """Structured array holding the measured seconds and femtoseconds delay values in two columns."""
        arr: NDArray = np.column_stack((self.__seconds, self.__femto))
        struct_arr: NDArray = unstructured_to_structured(arr, dtype=np.dtype(DATA_TYPE))
        check_data_structure(struct_arr)
        return struct_arr

    @classmethod
    def from_path(cls, data_path: Path, seconds_add: int = 0) -> "NPETData":
        """
        Import NPET data from a file.
        :param data_path: Path to the data file
        :param seconds_add: Number of seconds to add to each epoch, can be positive or negative
        :return: NPETData object
        """
        data: NDArray = import_data(data_path, seconds_add)
        return cls(__seconds=data["seconds"], __femto=data["femto"])

    def adjust_seconds(self, seconds_add: int) -> None:
        """
        Adjust the `seconds` delay values by a given amount.
        :param seconds_add: The number of seconds to add to each epoch; can be positive or negative
        """
        self.__seconds += seconds_add

    def discard_rows_until_ref_match(self, data_ref: NDArray) -> None:
        """
        Discard rows from the data until the first row of the first column matches the reference data.
        :param data_ref: Reference data, the first row of the first column is used to match.
        """
        new_data, discarded = discard_rows_until_first_col_match(
            data_to_process=self.structured_arr,
            data_ref=data_ref,
        )
        if discarded > 0:
            self.__seconds = new_data["seconds"]
            self.__femto = new_data["femto"]

    def calc_delay_start(self, *, stop: NDArray, frequency: int) -> "NPETData":
        """
        Calculate the delay between this data (start) and the given stop data.
        :param stop: Stop data to calculate the delay against
        :param frequency: Frequency of the data
        :return: NPETData of the calculated delays
        """
        ret = calculate_delay(
            data_start=self.structured_arr,
            data_stop=stop,
            frequency=frequency,
        )
        return NPETData(__seconds=ret["seconds"], __femto=ret["femto"])

    def calc_delay_stop(self, *, start: NDArray, frequency: int) -> "NPETData":
        """
        Calculate the delay between the given start data and this data (stop).
        :param start: Start data to calculate the delay against
        :param frequency: Frequency of the data
        :return: NPETData of the calculated delays
        """
        ret = calculate_delay(
            data_start=start,
            data_stop=self.structured_arr,
            frequency=frequency,
        )
        return NPETData(__seconds=ret["seconds"], __femto=ret["femto"])

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
            self.__femto,
            bin_size=bin_size,
            percentage_threshold=percentage_threshold,
        )

    def define_signal_range(
        self, signal_mask: NDArray[np.bool_]
    ) -> tuple[float, float]:
        """
        Calculate a range of data around the detected signal.
        :param signal_mask: Boolean mask indicating the detected signal
        :return: A tuple defining the range min and max values
        """
        signal_values: NDArray[np.int_] = self.__femto[signal_mask]
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
        filtered_seconds = self.__seconds[filter_mask]
        filtered_femto = self.__femto[filter_mask]
        return NPETData(__seconds=filtered_seconds, __femto=filtered_femto)

    def recursive_sigma_filter(
        self,
        sigma_mult: float,
        max_iter: int = 100,
    ) -> "NPETData":
        """
        Perform recursive sigma filtering on the data.
        :param sigma_mult: The sigma multiplier for the filtering
        :param max_iter: The maximum number of iterations
        :return: NPETData object with filtered data
        """
        res, _ = recursive_sigma_filter(
            self.structured_arr,
            sigma_mult=sigma_mult,
            max_iter=max_iter,
        )
        return NPETData(__seconds=res["seconds"], __femto=res["femto"])
