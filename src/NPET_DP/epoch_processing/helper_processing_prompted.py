import typer
from click import Choice
from numpy.typing import NDArray

from NPET_DP.epoch_processing.helper_funcs import check_data_structure, validate_inputs
from NPET_DP.epoch_processing.helper_processing import remove_drift


@validate_inputs
def drift_removal_prompt(data: NDArray) -> tuple[NDArray, int]:
    """
    Prompt the user for optional drift removal from data.
    :param data: Data to remove drift from
    :return: Data with drift removed, if applicable by the user input.
    """
    pol_deg: int = typer.prompt(
        "Enter the polynomial degree for drift removal (0 - no drift compensation)",
        type=Choice([0, 1, 2]),
        default=0,
        show_choices=False,
    )
    if data.shape[0] < pol_deg + 1:
        typer.secho(
            f"Required >={pol_deg + 1} points in data, skipping drift removal ...",
            fg=typer.colors.RED,
        )
        return data, 0
    if pol_deg == 0:
        return data, 0
    typer.echo(f"Removing drift with polynomial degree {pol_deg} ...")
    no_drift = remove_drift(data, deg=pol_deg)
    check_data_structure(no_drift)
    return no_drift, pol_deg
