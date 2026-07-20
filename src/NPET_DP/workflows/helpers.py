import typer
from click import Choice

from NPET_DP.processing.data_struct import NPETData


def drift_removal_prompt(data: NPETData) -> tuple[NPETData, int]:
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
    if len(data) < pol_deg + 1:
        typer.secho(
            f"Required >={pol_deg + 1} points in data, skipping drift removal ...",
            fg=typer.colors.RED,
        )
        return data, 0
    if pol_deg == 0:
        return data, 0
    typer.echo(f"Removing drift with polynomial degree {pol_deg} ...")
    return data.compensate_drift(pol_deg), pol_deg
