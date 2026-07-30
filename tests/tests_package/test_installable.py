import subprocess
import tomllib
import venv
from contextlib import suppress
from pathlib import Path

import pytest
from pytest import FixtureRequest
from tests_package.conftest import get_system_specific_pypath


@pytest.fixture(scope="session")
def no_deps_venv_from_wheel(
    tmp_path_factory: pytest.TempPathFactory,
    wheel_file: Path,
) -> Path:
    """Install NPET_DP from .whl file.

    Create venv and install NPET_DP in it from the wheel.
    Dependencies are not installed.
    """
    assert wheel_file.is_file(), f"Wheel file not found: {wheel_file}"
    venv_dir: Path = tmp_path_factory.mktemp("venv")
    venv.create(venv_dir, with_pip=True)
    python_bin: Path = get_system_specific_pypath(venv_dir)
    subprocess.run(  # noqa: S603
        [str(python_bin), "-m", "pip", "install", "--no-deps", wheel_file],
        check=True,
    )
    return python_bin


@pytest.fixture(scope="session")
def no_deps_venv_from_git(
    tmp_path_factory: pytest.TempPathFactory,
    tests_root: Path,
) -> Path:
    """Install NPET_DP from git repo.

    Create venv and install NPET_DP in it directly from the project's GitHub URL,
    as declared under ``project.url`` in pyproject.toml. Dependencies are not installed.
    """
    pyproject_file: Path = tests_root.parent / "pyproject.toml"
    repo_url: str = tomllib.loads(pyproject_file.read_text())["project"]["url"]
    venv_dir: Path = tmp_path_factory.mktemp("venv")
    venv.create(venv_dir, with_pip=True)
    python_bin: Path = get_system_specific_pypath(venv_dir)
    subprocess.run(  # noqa: S603
        [str(python_bin), "-m", "pip", "install", "--no-deps", f"git+{repo_url}"],
        check=True,
    )
    return python_bin


@pytest.fixture(scope="session")
def python_path(request: FixtureRequest) -> Path:
    """Get the path to the python executable.

    The python executable is located within the venv directory,
    where the package was installed.
    """
    # noinspection PyUnresolvedReferences
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def path_to_package(python_path: Path) -> Path:
    """Get the path to the package files.

    The package files are located within the venv directory,
    where the package was installed.
    The path is determined by importing the package and getting the
    __file__ attribute, which points to the location of the package.
    """
    res = subprocess.run(  # noqa: S603
        [str(python_path), "-c", "import NPET_DP; print(NPET_DP.__file__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Failed to get path to package: {res.stderr}"
    path: Path = Path(res.stdout.strip()).parent
    assert path.is_dir(), f"Path to package is not a file: {path}"
    return path


@pytest.mark.smoke
@pytest.mark.parametrize(
    "python_path",
    ["no_deps_venv_from_wheel", "no_deps_venv_from_git"],
    indirect=True,
)
@pytest.mark.xdist_group(name="package")
def test_package_is_installed(python_path: Path) -> None:
    """Test that the package is installed.

    Test that package was successfully installed in the venv.
    Check that pip can show information about the package.
    """
    res = subprocess.run(  # noqa: S603
        [str(python_path), "-m", "pip", "show", "NPET_DP"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Failed to show package: {res.stderr}"
    assert "Name: NPET_DP" in res.stdout, f"Package not found: {res.stdout}"


@pytest.mark.parametrize(
    "python_path",
    ["no_deps_venv_from_wheel", "no_deps_venv_from_git"],
    indirect=True,
)
@pytest.mark.xdist_group(name="package")
def test_only_npet_dp_installed(python_path: Path) -> None:
    """Test that only NPET_DP is installed.

    Test that no other packages are installed in the venv, except for pip and setuptools,
    which might come preloaded.
    """
    res = subprocess.run(  # noqa: S603
        [str(python_path), "-m", "pip", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Failed to list packages: {res.stderr}"
    installed: list[str] = [line.split()[0] for line in res.stdout.splitlines()[2:]]
    with suppress(ValueError):
        installed.remove("pip")
        installed.remove("setuptools")
    assert installed == ["NPET_DP"], f"Unexpected packages installed: {installed}"


@pytest.mark.parametrize(
    "python_path",
    ["no_deps_venv_from_wheel", "no_deps_venv_from_git"],
    indirect=True,
)
@pytest.mark.xdist_group(name="package")
def test_version_included(path_to_package: Path) -> None:
    """Test that the version file is included in the package.

    Test that the artifact file containing the current cersion of the pacakage,
    is included in the package.
    """
    version_file: Path = path_to_package / "_version.py"
    assert version_file.is_file(), f"Version file not found: {version_file}"
    contents: str = version_file.read_text()
    assert "__version__" in contents, f"Version not found in file: {version_file}"
