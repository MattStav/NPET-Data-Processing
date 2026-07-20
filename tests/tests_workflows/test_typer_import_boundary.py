import ast
from pathlib import Path

import NPET_DP


def _imports_typer(source_file: Path) -> bool:
    """Check whether a Python source file imports the typer package."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "typer" or alias.name.startswith("typer.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "typer" or node.module.startswith("typer.")):
                return True
    return False


def test_typer_only_imported_in_root_workflows_and_framework_modules():
    """Test that typer is only imported directly in the package root or in the `workflows`/`framework` modules."""
    # noinspection PyTypeChecker
    package_root: Path = Path(NPET_DP.__file__).parent
    workflows_dir: Path = package_root / "workflows"
    framework_dir: Path = package_root / "framework"
    offenders: list[Path] = []
    for source_file in package_root.rglob("*.py"):
        if (
            source_file.parent != package_root
            and workflows_dir not in source_file.parents
            and framework_dir not in source_file.parents
        ):
            if _imports_typer(source_file):
                offenders.append(source_file.relative_to(package_root))
    assert not offenders, f"typer must only be imported in the NPET_DP root, workflows, or framework modules, but found it imported in: {offenders}"
