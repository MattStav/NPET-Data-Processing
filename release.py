from typing import Literal, get_args
import subprocess

import typer
from click import Choice  # Included in typer


"""
This script is used to semi-automate the release process.
It uses uv to build the wheel and git to create and push the tag.
"""

_VERSIONS_TYPE = Literal["major", "minor", "patch"]
_VERSIONS = get_args(_VERSIONS_TYPE)
_col = typer.colors


def __run_command(cmd: list[str], check: bool = True) -> str:
    """
    Runs a shell cmd and returns the output.
    :param cmd: The command to run.
    :param check: Whether to check the return code. If True, an exception is raised if the command fails.
    :return: The output of the command.
    """
    typer.echo(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    return result.stdout.strip()


def __get_latest_version() -> str:
    """Gets the latest version using git describe."""
    cmd: list[str] = ["git", "describe", "--tags", "--abbrev=0"]
    tag: str = __run_command(cmd, check=False).lstrip("v")
    if not tag:
        typer.secho("No existing tags found, starting from 0.0.0", fg=_col.YELLOW)
        return "0.0.0"
    return tag


def __increment_version(version: str, update_type: _VERSIONS_TYPE) -> str:
    """
    Increments the version based on the update type.
    :param version: The current version.
    :param update_type: The type of update (major, minor, patch).
    :return: The new version.
    """
    assert update_type in _VERSIONS, f"Invalid update type: {update_type}"
    parts: list[int] = list(map(int, version.split(".")[:3]))
    while len(parts) < 3:
        parts.append(0)
    if update_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif update_type == "minor":
        parts[1] += 1
        parts[2] = 0
    elif update_type == "patch":
        parts[2] += 1
    return f"{parts[0]}.{parts[1]}.{parts[2]}"


def main():
    """Main function to run the release process."""
    typer.secho("Starting release process...", fg=_col.CYAN)
    if __run_command(["git", "status", "--short"]):
        typer.secho("There are uncommitted changes in the repository!", fg=_col.YELLOW)
        typer.confirm("Do you want to proceed with the release?", abort=True)
    current_version: str = __get_latest_version()
    typer.echo(f"Current version: {current_version}")
    # Prompt for the update type
    update_type: _VERSIONS_TYPE = typer.prompt(
        "Enter update type",
        show_choices=True,
        type=Choice(_VERSIONS, case_sensitive=False),
    )
    # Calculate new version
    new_version: str = __increment_version(current_version, update_type)
    new_tag: str = f"v{new_version}"
    typer.secho(f"New version will be: {new_version} (Tag: {new_tag})", fg=_col.GREEN)
    # Confirm before proceeding
    typer.confirm(f"Create and push tag {new_tag}, and build new wheel?", abort=True)
    __run_command(["git", "tag", "-a", new_tag, "-m", f"Release {new_tag}"])
    typer.secho(f"Created tag {new_tag}", fg=_col.GREEN)
    try:
        __run_command(["git", "push", "origin", new_tag])
        typer.secho(f"Pushed tag {new_tag} to remote", fg=_col.GREEN)
    except Exception as e:
        typer.echo(f"Failed to push tag: {e}", err=True)
        typer.secho("Tag remains locally! DELETE IT!", fg=_col.RED)
        raise typer.Exit(1)
    typer.echo(f"Building wheel for {new_tag} ...")
    __run_command(["uv", "build", "--wheel"])
    typer.secho("Build completed successfully", fg=_col.GREEN)


if __name__ == "__main__":
    main()
