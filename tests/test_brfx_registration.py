import json
import os
import subprocess
from pathlib import Path
from typing import cast

ROOT = Path(__file__).parents[1]
PROJECT = ROOT / "project.yml"
DESIGN = ROOT / "design" / "system.yml"


def _peep_color(registry: Path, appearance: str) -> dict[str, object]:
    environment = os.environ | {"BRFX_REGISTRY_PATH": str(registry)}
    result = subprocess.run(
        ["brfx", "peep", "color", "sibyl", "accent.primary", "--theme", "sibyl",
         "--appearance", appearance, "--format", "json"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


def test_sibyl_metadata_uses_normal_brfx_registration_shape() -> None:
    project = PROJECT.read_text()
    assert "name: sibyl" in project
    assert "design: design/system.yml" in project
    assert "theme: sibyl" in project
    assert "executable: bin/sibyl" in project


def test_sibyl_identity_has_distinct_semantic_light_and_dark_accents(tmp_path: Path) -> None:
    registry = tmp_path / "projects.json"
    registry.write_text(json.dumps({"version": 1, "projects": {"sibyl": {"path": str(ROOT)}}}))

    light = _peep_color(registry, "light")
    dark = _peep_color(registry, "dark")

    assert light["theme"] == "sibyl"
    assert light["semantic"] == "accent.primary"
    assert light["appearance"] == "light"
    assert light["hex"] == "#146A9C"
    assert dark["appearance"] == "dark"
    assert dark["hex"] == "#1FB3FF"
    assert str(light["hex"]) != str(dark["hex"])


def test_identity_values_are_not_repeated_in_sibyl_runtime_or_outputs() -> None:
    runtime_files = list((ROOT / "src").rglob("*.py"))
    runtime = "\n".join(path.read_text() for path in runtime_files)
    assert "#1FB3FF" not in runtime
    assert "#146A9C" not in runtime
    assert "brfx" not in runtime.lower()

    output_contract = (ROOT / "src" / "sibyl" / "transform.py").read_text().lower()
    assert "design bench" not in output_contract
    assert "project.yml" not in output_contract
