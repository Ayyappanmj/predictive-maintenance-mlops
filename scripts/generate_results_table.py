"""Turn models/metadata.json into a Markdown results section (no numbers are typed by hand).

Usage:
  python scripts/generate_results_table.py                 # print to the terminal
  python scripts/generate_results_table.py --update-readme # replace the README results block
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
        f"{m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['threshold']:.3f} |"
    )


def results_markdown(meta: dict) -> str:
    t = meta["test_metrics"]
    cm = t["confusion_matrix"]
    lines = [
        f"Model version `{meta['model_version']}` (trained {meta['trained_at_utc'][:10]}, "
        f"seed {meta['seed']}, selected: **{meta['model_name']}**).",
        "",
        "| Model / split | Precision | Recall | F1 | PR-AUC | ROC-AUC | Threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in meta["validation_metrics"].items():
        lines.append(_row(f"{name} (validation)", m))
    lines.append(_row(f"**{meta['model_name']} (test, final)**", t))
    lines += [
        "",
        f"Test set: {t['n_samples']} rows, {t['n_positive']} failures. Confusion matrix "
        f"(TN/FP/FN/TP): {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']}.",
        "The model was chosen on validation PR-AUC; the test set was used once afterwards. "
        "With few failures in the test set these figures are noisy. The data is synthetic.",
    ]
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--update-readme", action="store_true")
    a = ap.parse_args()
    meta_path = ROOT / a.model_dir / "metadata.json"
    if not meta_path.exists():
        print(f"{meta_path} not found. Train a model first: python -m src.train")
        return 1
    block = results_markdown(json.loads(meta_path.read_text()))
    if a.update_readme:
        update_readme(ROOT / "README.md", block)
        print("README.md results block updated.")
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
