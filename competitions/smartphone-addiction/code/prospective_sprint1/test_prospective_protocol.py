"""Static protocol guards for the prospective S6E8 sprint."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def calls_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return calls


def test_generation_does_not_read_holdout_results() -> None:
    text = (ROOT / "train_selected_xgb.py").read_text(encoding="utf-8")
    assert "sealed_holdout_results.json" not in text
    assert '"holdout_scored": False' in text
    # Its only AUC calls are on valid_mask/development, never holdout.
    auc_lines = [line.strip() for line in text.splitlines() if "roc_auc_score" in line and "import" not in line]
    assert auc_lines
    assert all("holdout" not in line for line in auc_lines)


def test_ablation_marks_holdout_unscored() -> None:
    text = (ROOT / "experiment_prospective_te.py").read_text(encoding="utf-8")
    assert '"holdout_scored": False' in text
    assert "y_all[holdout" not in text


def test_evaluation_has_one_time_guard() -> None:
    text = (ROOT / "select_prospective_blend.py").read_text(encoding="utf-8")
    assert 'if spec.get("holdout_opened")' in text
    assert 'spec["holdout_opened"] = True' in text


if __name__ == "__main__":
    test_generation_does_not_read_holdout_results()
    test_ablation_marks_holdout_unscored()
    test_evaluation_has_one_time_guard()
    for filename in [
        "experiment_prospective_te.py", "train_selected_xgb.py",
        "select_prospective_blend.py", "gen_prospective_realmlp_nb.py",
    ]:
        ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    print("prospective protocol guards: OK")
