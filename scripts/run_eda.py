"""Exploratory data analysis. Writes reports/eda_summary.json and PNG charts.

Usage: python scripts/run_eda.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_ingestion import ROOT, load_config, load_raw  # noqa: E402
from src.data_validation import validate_dataframe  # noqa: E402


def main() -> int:
    cfg = load_config()
    df = load_raw(cfg)
    rep = validate_dataframe(df, cfg)
    if not rep.ok:
        print("Validation failed:", rep.errors)
        return 1
    d = cfg["data"]
    num, tgt = d["numeric_features"], d["target"]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)

    summary = {
        "n_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df[tgt].value_counts().items()},
        "failure_rate_by_type": df.groupby("Type")[tgt].mean().round(4).to_dict(),
        "numeric_describe": df[num].describe().round(3).to_dict(),
        "correlation_with_target": df[[*num, tgt]].corr()[tgt].drop(tgt).round(3).to_dict(),
        "validation": rep.to_dict(),
        "note": "Dataset is synthetic (AI4I 2020, UCI, CC BY 4.0).",
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), num, strict=False):
        for label, g in df.groupby(tgt):
            ax.hist(
                g[col],
                bins=30,
                alpha=0.6,
                density=True,
                label="failure" if label == 1 else "no failure",
            )
        ax.set_title(col)
        ax.legend()
    df[tgt].value_counts().sort_index().plot.bar(ax=axes.ravel()[-1], color=["#4c78a8", "#e45756"])
    axes.ravel()[-1].set_title("Class balance (0 = no failure, 1 = failure)")
    fig.tight_layout()
    fig.savefig(out / "eda_distributions.png", dpi=120)
    print("Wrote reports/eda_summary.json and reports/eda_distributions.png")
    print(json.dumps(summary["class_counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
