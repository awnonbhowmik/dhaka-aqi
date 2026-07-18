#!/usr/bin/env python3
"""Extract committed notebook outputs without re-running legacy analysis."""

from __future__ import annotations

import base64
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "original_results_snapshot"


def _text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def snapshot_notebook(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    stem_dir = OUT / path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = [f"# Stored outputs: {path.name}", ""]
    image_count = 0
    error_count = 0
    executed = 0

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            executed += 1
        outputs = cell.get("outputs", [])
        if not outputs:
            continue
        report.extend([f"## Cell {index}", ""])
        for output in outputs:
            output_type = output.get("output_type")
            if output_type == "error":
                error_count += 1
                report.append(
                    f"ERROR {output.get('ename')}: {output.get('evalue')}"
                )
            elif output_type == "stream":
                report.append("```text\n" + _text(output.get("text", "")) + "\n```")
            else:
                data = output.get("data", {})
                if "text/plain" in data:
                    report.append("```text\n" + _text(data["text/plain"]) + "\n```")
                if "text/html" in data:
                    report.append(_text(data["text/html"]))
                if "image/png" in data:
                    image_count += 1
                    filename = f"cell_{index:03d}_{image_count:03d}.png"
                    (stem_dir / filename).write_bytes(
                        base64.b64decode(_text(data["image/png"]))
                    )
                    report.append(f"Stored image: `{filename}`")
        report.append("")

    (stem_dir / "outputs.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "path": path.name,
        "code_cells": sum(
            c.get("cell_type") == "code" for c in notebook.get("cells", [])
        ),
        "executed_code_cells": executed,
        "stored_errors": error_count,
        "stored_png_images": image_count,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = [snapshot_notebook(ROOT / name) for name in ("main.ipynb", "analysis.ipynb")]
    freeze = subprocess.run(
        ["python3", "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    (OUT / "package_versions.txt").write_text(
        f"Python {platform.python_version()}\n" + freeze.stdout,
        encoding="utf-8",
    )
    summary = {
        "starting_commit": "deb9b0dc064e9a6603f76415b50aa8f69fb394cf",
        "snapshot_method": "stored_notebook_outputs_not_reexecuted",
        "notebooks": notebooks,
        "manuscript_available": False,
        "known_legacy_study_period": "2017-01 through 2025-12",
        "known_legacy_daily_period": "2017-01-01 through 2026-05-02",
    }
    (ROOT / "reports" / "original_key_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

