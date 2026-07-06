from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Get the root directory of the tests."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(tests_root: Path) -> Path:
    """Get the path to the data directory."""
    return tests_root / "test_data"
