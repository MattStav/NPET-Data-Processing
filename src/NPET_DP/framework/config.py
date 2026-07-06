from pathlib import Path
from typing import Literal, Optional

import typer
from pydantic import BaseModel, ConfigDict, PrivateAttr


class __AppConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    """Configuration holder for application settings"""
    # Directory where source data is accessed
    _input_data_dir: Path = PrivateAttr(default=Path.cwd())
    # Data gathering frequency used in the processing
    _frequency: Optional[int] = PrivateAttr(default=None)
    # Minimum delay in ns by which to filter the data
    _min_delay: Optional[float] = PrivateAttr(default=None)
    # Maximum delay in ns by which to filter the data
    _max_delay: Optional[float] = PrivateAttr(default=None)
    # Sigma value for the gaussian filter
    _sigma: Optional[float] = PrivateAttr(default=None)

    @property
    def input_data_dir(self) -> Path:
        if not self._input_data_dir or not self._input_data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self._input_data_dir}")
        return self._input_data_dir

    @input_data_dir.setter
    def input_data_dir(self, new_path: Optional[Path]) -> None:
        """
        Validate the data directory path.
        When None is entered, the user is prompted in CLI to enter a new path.
        :param new_path: Path to a new data directory.
        :return: New data directory path.
        """
        while not new_path:
            typer.echo(f"Current data path: {self._input_data_dir}")
            temp_path: Path = Path(typer.prompt("Insert new data path", type=str))
            # If the path is not absolute, append it to the current data dir
            if not temp_path.is_absolute() and self._input_data_dir:
                temp_path: Path = self._input_data_dir / temp_path
            # Keep iterating until the path exists and is a directory
            if temp_path.is_dir():
                new_path = temp_path.resolve()
                break
            typer.secho(f"Invalid path: {temp_path}", fg=typer.colors.RED)
        assert new_path is not None, "New path should not be None"
        # This is triggered if the supplied path does not exist
        if not new_path.is_dir():
            typer.secho(f"Dir not found: {new_path}", err=True)
            raise FileNotFoundError(f"Dir not found: {new_path}")
        typer.echo(f"Using data dir: {new_path}")
        self._input_data_dir = new_path

    @property
    def frequency(self) -> int:
        if not self._frequency:
            self.prompt_frequency()
        assert self._frequency
        typer.echo(f"Current data gathering frequency: {self._frequency} Hz")
        return self._frequency

    @frequency.setter
    def frequency(self, new_frequency: int) -> None:
        if not isinstance(new_frequency, int) or new_frequency <= 0:
            raise ValueError("Frequency must be positive")
        self._frequency = new_frequency

    def prompt_frequency(self) -> None:
        """Prompt the user to enter a frequency value, which is saved to config"""
        typer.echo(f"Current data gathering frequency: {self._frequency} Hz")
        while True:
            try:
                self.frequency = typer.prompt(
                    "Insert the data gathering frequency [Hz]",
                    type=int,
                )
                return
            except ValueError:
                typer.secho("Invalid frequency!", fg=typer.colors.RED)

    @property
    def sigma(self) -> float:
        if not self._sigma:
            self.prompt_sigma()
        assert self._sigma
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma: float) -> None:
        if not isinstance(new_sigma, float) or new_sigma <= 0:
            raise ValueError("Sigma must be positive")
        self._sigma = new_sigma

    def prompt_sigma(self) -> None:
        """Prompt the user to enter a sigma value, which is saved to config"""
        typer.echo(f"Current sigma filtering: {self._sigma}")
        while True:
            try:
                self.sigma = typer.prompt(
                    "Insert the sigma value for the gaussian filter",
                    type=float,
                    default=2.2,
                )
                return
            except ValueError:
                typer.secho("Invalid sigma value!", fg=typer.colors.RED)

    @property
    def min_delay(self) -> float:
        if not self._min_delay:
            typer.echo("Minimum delay filter value not set. Setting it now...")
            self.prompt_delay("min")
        assert self._min_delay is not None
        return self._min_delay

    @min_delay.setter
    def min_delay(self, new_min_delay: float) -> None:
        assert isinstance(new_min_delay, float)
        self._min_delay = new_min_delay

    @property
    def max_delay(self) -> float:
        if not self._max_delay:
            typer.echo("Maximum delay filter value not set. Setting it now...")
            self.prompt_delay("max")
        assert self._max_delay is not None
        return self._max_delay

    @max_delay.setter
    def max_delay(self, new_max_delay: float) -> None:
        assert isinstance(new_max_delay, float)
        self._max_delay = new_max_delay

    def prompt_delay(
        self,
        delay_type: Literal["min", "max"],
        validate: bool = True,
    ) -> None:
        """
        Prompt the user to enter a delay value.
        :param delay_type: The type of delay to prompt for.
        :param validate: Whether to validate the delay value.
        If enabled, the max delay must be greater than the min delay.
        """

        def validate_delay(val: float) -> bool:
            """Validate the delay value."""
            if delay_type == "min":
                return self._max_delay is None or val * mult < self._max_delay
            if delay_type == "max":
                return self._min_delay is None or val * mult > self._min_delay
            return True

        mult: float = 1e6  # ns to fs
        attr = getattr(self, f"_{delay_type}_delay")
        while True:
            new_val = typer.prompt(
                f"Insert the {delay_type.capitalize()} delay filter value [ns]",
                type=float,
                default=0 if attr is None else round(attr / mult, 5),
            )
            if not validate or validate_delay(new_val):
                setattr(self, f"_{delay_type}_delay", new_val * mult)
                break
            typer.secho("Invalid delay value!", fg=typer.colors.RED)


config: __AppConfig = __AppConfig()
