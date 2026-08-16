"""Dependency-free repository quality checks for CI and local use."""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
SECRET_PATTERNS = (
    re.compile("KG" + r"AT_[A-Za-z0-9]+"),
    re.compile(
        r"(api[_-]?key|password|secret)\s*=\s*['\"][^'\"]{6,}['\"]",
        re.IGNORECASE,
    ),
)


def included(path: pathlib.Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.parts)


def repository_files() -> list[pathlib.Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and included(path)]


def main() -> int:
    failures: list[str] = []

    python_files = [p for p in ROOT.rglob("*.py") if included(p)]
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # noqa: BLE001 - aggregate all repository failures
            failures.append(f"Python invalide: {path.relative_to(ROOT)}: {exc}")

    notebooks = [p for p in ROOT.rglob("*.ipynb") if included(p)]
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                raise ValueError("notebook v4/cells attendus")
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)
                    ast.parse(source, filename=f"{path}#cell-{index}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Notebook invalide: {path.relative_to(ROOT)}: {exc}")

    files = repository_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            failures.append(f"Lecture impossible: {path.relative_to(ROOT)}: {exc}")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                failures.append(
                    f"Secret potentiel: {path.relative_to(ROOT)}:{line_number}"
                )

    # experiments.csv : structure uniforme + experiment_id unique
    import csv
    csv_path = ROOT / "experiments.csv"
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if rows:
            n_cols = len(rows[0])
            for i, row in enumerate(rows):
                if len(row) != n_cols:
                    failures.append(
                        f"experiments.csv ligne {i}: {len(row)} colonnes != {n_cols}")
            ids = [r[0] for r in rows[1:] if r and r[0]]
            if len(ids) != len(set(ids)):
                failures.append("experiments.csv: experiment_id non unique")
    except OSError as exc:
        failures.append(f"experiments.csv illisible: {exc}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    print(
        f"OK: {len(python_files)} fichiers Python, {len(notebooks)} notebooks, "
        f"{len(files)} fichiers du dépôt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
