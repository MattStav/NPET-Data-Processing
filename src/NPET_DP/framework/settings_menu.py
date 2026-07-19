import typer

from NPET_DP.framework.config import config


def settings_menu() -> None:
    """Show a settings menu"""
    typer.secho("\n========= Settings Menu =========", bold=True)
    typer.echo("\t1. Data gathering frequency")
    typer.echo("\t2. Sigma filter")
    typer.echo("\t3. Source dir path")
    typer.echo("\t0. Return to main menu")

    user_choice: int = typer.prompt("Select setting to adjust", type=int)
    match user_choice:
        case 1:
            config.prompt_frequency()
        case 2:
            config.prompt_sigma()
        case 3:
            config.input_data_dir = None
        case 0:
            return
        case _:
            typer.secho("Invalid choice", fg=typer.colors.RED)
            return
