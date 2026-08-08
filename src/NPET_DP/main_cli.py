import sys
from pathlib import Path
from typing import Annotated

import typer
from matplotlib import pyplot as plt

from NPET_DP._version import __version__
from NPET_DP.framework.config import config
from NPET_DP.framework.constants import APP_NAME
from NPET_DP.framework.path_handler import open_plot_outputs
from NPET_DP.framework.settings_menu import settings_menu
from NPET_DP.workflows.one_epoch import main_one_epoch
from NPET_DP.workflows.pps import main_pps
from NPET_DP.workflows.two_epochs import main_two_epochs

# Define the app properties
npet_dp = typer.Typer(
    name=APP_NAME,
    add_completion=False,
    help="Package for processing NPET data",
    epilog="Run without a subcommand to launch the interactive menu.",
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


def _version_callback(callback_enabled: bool) -> None:
    """Print the version of the package, when enabled.

    :param callback_enabled: Whether the callback is enabled.
    """
    if not callback_enabled:
        return
    print(f"{APP_NAME} {__version__}")
    raise typer.Exit()


@npet_dp.callback(invoke_without_command=True)
def arg_parse(
    ctx: typer.Context,
    source_data_dir: str = typer.Option(
        str(Path.cwd()),
        "--data-path",
        "-dp",
        help="Path to directory with the NPET data to process",
    ),
    ver: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Parse top-level CLI options and launch the interactive menu if no subcommand given."""
    typer.secho(
        f"{APP_NAME} launched: v{__version__}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    try:
        config.input_data_dir = Path(source_data_dir).resolve()
    except FileNotFoundError:
        # DO NOT CHANGE THIS EXIT CODE
        # NPET_communication_FW expects this exit code
        sys.exit(10)
    if ctx.invoked_subcommand is None:
        ctx.invoke(main_menu)


@npet_dp.command()
def main_menu() -> None:
    """Show a simple interactive menu."""
    while True:
        typer.secho("\n========= Main Menu =========", bold=True)
        typer.echo("1. Process Single Epoch")
        typer.echo("2. Process Dual Epochs")
        typer.echo("3. Process PPS")
        typer.echo("4. Settings")
        typer.echo("5. Open Outputs")
        typer.echo("0. Exit")
        user_choice: int = typer.prompt("Select a menu item", type=int)
        match user_choice:
            case 1:
                typer.secho("\nProcessing single epoch data", fg=typer.colors.CYAN)
                plt.close("all")
                main_one_epoch()
            case 2:
                typer.secho("\nProcessing dual epochs data", fg=typer.colors.CYAN)
                plt.close("all")
                main_two_epochs()
            case 3:
                typer.secho("\nProcessing PPS data", fg=typer.colors.CYAN)
                plt.close("all")
                main_pps()
            case 4:
                settings_menu()
            case 5:
                typer.echo("\nOpening outputs folder ...")
                open_plot_outputs()
            case 0:
                typer.echo(f"Program {APP_NAME} terminated")
                return
            case _:
                typer.secho("Invalid choice", fg=typer.colors.RED)
