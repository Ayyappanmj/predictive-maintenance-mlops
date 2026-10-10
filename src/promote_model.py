"""Safe model promotion, archiving and rollback.

A candidate model replaces the approved model ONLY if every gate passes. Gates use VALIDATION
metrics (never the test set). On any failure the approved model is left untouched and the
reasons are appended to model_registry/promotion_log.jsonl.

Layout:  models/                          approved model served by the API
         model_registry/candidate/        freshly trained candidate (CI working dir)
         model_registry/archive/<ver>/    previously approved models (for rollback)
         model_registry/promotion_log.jsonl

CLI:  python -m src.promote_model promote --tests-passed true   (add --dry-run to preview)
      python -m src.promote_model rollback --version 20260101-030000
      python -m src.promote_model list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.api_contract import COLUMN_MAP, EXAMPLE_READING
from src.data_ingestion import ROOT, load_config
from src.predict import load_model, predict_dataframe
from src.preprocessing import input_columns

FILES = ("model.joblib", "metadata.json", "reference.csv")
INSTALL_ORDER = ("reference.csv", "model.joblib", "metadata.json")  # metadata last
DEFAULT_APPROVED = ROOT / "models"
DEFAULT_REGISTRY = ROOT / "model_registry"
REQUIRED_META = (
    "model_version",
    "model_name",
    "trained_at_utc",
    "decision_threshold",
    "risk_thresholds",
    "input_schema",
    "validation_metrics",
    "test_metrics",
    "data_sha256",
    "model_sha256",
    "dataset_note",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_meta(model_dir) -> dict | None:
    p = Path(model_dir) / "metadata.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def has_model(model_dir) -> bool:
    return (Path(model_dir) / "model.joblib").exists() and read_meta(model_dir) is not None


def check_artifact_integrity(model_dir) -> tuple[bool, str]:
    d = Path(model_dir)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        return False, f"missing files: {missing}"
    meta = read_meta(d)
    if meta is None:
        return False, "metadata.json is not valid JSON"
    if sha256_file(d / "model.joblib") != meta.get("model_sha256"):
        return False, "model.joblib sha256 does not match metadata"
    return True, "model.joblib sha256 matches metadata"


def _val(meta: dict, key: str) -> float:
    return float(meta["validation_metrics"][meta["model_name"]][key])


def check_input_schema(meta: dict, cfg: dict) -> tuple[bool, str]:
    schema = meta.get("input_schema", {})
    expected = input_columns(cfg)
    if schema.get("columns") != expected:
        return False, f"input columns {schema.get('columns')} != expected {expected}"
    return True, "input schema matches the API and configuration"


def check_api_compatibility(model_dir, cfg: dict) -> tuple[bool, str]:
    """Load the candidate and run one prediction through the same path the API uses."""
    try:
        meta = read_meta(model_dir) or {}
        absent = [k for k in REQUIRED_META if k not in meta]
        if absent:
            return False, f"metadata missing keys required by the API: {absent}"
        frame = pd.DataFrame([{COLUMN_MAP[k]: v for k, v in EXAMPLE_READING.items()}])
        out = predict_dataframe(load_model(model_dir), frame, cfg)
        p = float(out["failure_probability"].iloc[0])
        if not 0.0 <= p <= 1.0:
            return False, f"probability out of range: {p}"
        return True, "loads and predicts through the API code path"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def run_gates(
    candidate_dir,
    approved_dir,
    cfg: dict,
    tests_passed: bool,
    min_pr_auc: float | None = None,
    min_recall: float | None = None,
    min_improvement: float | None = None,
) -> list[Check]:
    p = cfg["promotion"]
    min_pr_auc = p["min_val_pr_auc"] if min_pr_auc is None else min_pr_auc
    min_recall = p["min_val_recall"] if min_recall is None else min_recall
    min_improvement = p["min_improvement"] if min_improvement is None else min_improvement
    checks: list[Check] = []

    ok, detail = check_artifact_integrity(candidate_dir)
    checks.append(Check("artifact_integrity", ok, detail))
    meta = read_meta(candidate_dir) if ok else None

    checks.append(
        Check(
            "automated_tests",
            bool(tests_passed),
            "test suite passed" if tests_passed else "test suite did not pass",
        )
    )

    if meta is None:
        for n in (
            "input_schema",
            "api_compatibility",
            "min_val_pr_auc",
            "min_val_recall",
            "better_than_approved",
        ):
            checks.append(Check(n, False, "skipped: candidate artifacts failed integrity check"))
        return checks

    ok, detail = check_input_schema(meta, cfg)
    checks.append(Check("input_schema", ok, detail))
    ok, detail = check_api_compatibility(candidate_dir, cfg)
    checks.append(Check("api_compatibility", ok, detail))

    try:
        pr, rec = _val(meta, "pr_auc"), _val(meta, "recall")
    except (KeyError, TypeError, ValueError) as e:
        checks += [
            Check("min_val_pr_auc", False, f"validation metrics unreadable: {e}"),
            Check("min_val_recall", False, "validation metrics unreadable"),
            Check("better_than_approved", False, "validation metrics unreadable"),
        ]
        return checks
    checks.append(
        Check(
            "min_val_pr_auc",
            pr >= min_pr_auc,
            f"validation PR-AUC {pr:.4f} (required >= {min_pr_auc})",
        )
    )
    checks.append(
        Check(
            "min_val_recall",
            rec >= min_recall,
            f"validation recall {rec:.4f} (required >= {min_recall})",
        )
    )

    approved_meta = read_meta(approved_dir) if has_model(approved_dir) else None
    if approved_meta is None:
        checks.append(Check("better_than_approved", True, "no approved model yet"))
    else:
        try:
            base = _val(approved_meta, "pr_auc")
            checks.append(
                Check(
                    "better_than_approved",
                    pr >= base + min_improvement,
                    f"candidate PR-AUC {pr:.4f} vs approved {base:.4f} "
                    f"(required improvement >= {min_improvement})",
                )
            )
        except (KeyError, TypeError, ValueError):
            checks.append(
                Check(
                    "better_than_approved",
                    True,
                    "approved model has no readable validation metrics",
                )
            )
    return checks


def _install(src_dir: Path, dest_dir: Path) -> None:
    """Copy model files into dest_dir; restore the previous files if anything fails."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=dest_dir))
    backup = Path(tempfile.mkdtemp(prefix=".backup-", dir=dest_dir))
    moved: list[str] = []
    try:
        for f in FILES:
            shutil.copy2(src_dir / f, staging / f)
        for f in INSTALL_ORDER:
            if (dest_dir / f).exists():
                shutil.copy2(dest_dir / f, backup / f)
            os.replace(staging / f, dest_dir / f)
            moved.append(f)
    except Exception:
        for f in moved:
            if (backup / f).exists():
                shutil.copy2(backup / f, dest_dir / f)
            else:
                (dest_dir / f).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _archive(approved_dir: Path, registry_dir: Path) -> Path | None:
    meta = read_meta(approved_dir)
    if meta is None or not has_model(approved_dir):
        return None
    base = Path(registry_dir) / "archive" / str(meta["model_version"])
    target, n = base, 0
    while target.exists():
        n += 1
        target = base.with_name(f"{base.name}-{n}")
    target.mkdir(parents=True)
    for f in FILES:
        if (approved_dir / f).exists():
            shutil.copy2(approved_dir / f, target / f)
    return target


def _log(registry_dir: Path, entry: dict) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    with open(registry_dir / "promotion_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def promote_candidate(
    candidate_dir,
    approved_dir=DEFAULT_APPROVED,
    registry_dir=DEFAULT_REGISTRY,
    cfg: dict | None = None,
    tests_passed: bool = False,
    min_pr_auc: float | None = None,
    min_recall: float | None = None,
    min_improvement: float | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = cfg or load_config()
    candidate_dir, approved_dir, registry_dir = (
        Path(candidate_dir),
        Path(approved_dir),
        Path(registry_dir),
    )
    checks = run_gates(
        candidate_dir, approved_dir, cfg, tests_passed, min_pr_auc, min_recall, min_improvement
    )
    promoted = all(c.passed for c in checks)
    cand_meta, appr_meta = read_meta(candidate_dir), read_meta(approved_dir)
    decision = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "promote",
        "dry_run": dry_run,
        "candidate_version": cand_meta.get("model_version") if cand_meta else None,
        "approved_version_before": appr_meta.get("model_version") if appr_meta else None,
        "promoted": False,
        "checks": [asdict(c) for c in checks],
        "rejection_reasons": [f"{c.name}: {c.detail}" for c in checks if not c.passed],
    }
    if promoted and not dry_run:
        _archive(approved_dir, registry_dir)
        _install(candidate_dir, approved_dir)
        decision["promoted"] = True
    elif promoted:
        decision["would_promote"] = True
    if not dry_run:
        _log(registry_dir, decision)
    return decision


def rollback(version: str, approved_dir=DEFAULT_APPROVED, registry_dir=DEFAULT_REGISTRY) -> dict:
    approved_dir, registry_dir = Path(approved_dir), Path(registry_dir)
    src = registry_dir / "archive" / version
    if not src.is_dir():
        raise FileNotFoundError(f"No archived model {version} in {registry_dir / 'archive'}")
    ok, detail = check_artifact_integrity(src)
    if not ok:
        raise ValueError(f"Archived model {version} failed integrity check: {detail}")
    _archive(approved_dir, registry_dir)  # keep what is being replaced
    _install(src, approved_dir)
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "rollback",
        "restored_version": version,
        "promoted": True,
    }
    _log(registry_dir, entry)
    return entry


def list_versions(approved_dir=DEFAULT_APPROVED, registry_dir=DEFAULT_REGISTRY) -> dict:
    meta = read_meta(approved_dir)
    arch = Path(registry_dir) / "archive"
    return {
        "approved": meta.get("model_version") if meta else None,
        "archived": sorted(p.name for p in arch.iterdir()) if arch.is_dir() else [],
    }


def _opt_float(s: str):
    return float(s) if s not in (None, "") else None


def _str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y")


def _write_github_outputs(decision: dict) -> None:
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"promoted={'true' if decision['promoted'] else 'false'}\n")
            f.write(f"candidate_version={decision['candidate_version']}\n")
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        rows = "\n".join(
            f"| {c['name']} | {'pass' if c['passed'] else 'FAIL'} | {c['detail']} |"
            for c in decision["checks"]
        )
        verdict = "PROMOTED" if decision["promoted"] else "REJECTED (approved model kept)"
        with open(summ, "a", encoding="utf-8") as f:
            f.write(
                f"### Model promotion: {verdict}\n\n| Check | Result | Detail |\n"
                f"|---|---|---|\n{rows}\n"
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--candidate-dir", default=str(DEFAULT_REGISTRY / "candidate"))
    p.add_argument("--approved-dir", default=str(DEFAULT_APPROVED))
    p.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    p.add_argument(
        "--tests-passed",
        type=_str2bool,
        default=False,
        help="true only if the test suite passed (default false: fail closed)",
    )
    p.add_argument("--min-val-pr-auc", type=_opt_float, default=None)
    p.add_argument("--min-val-recall", type=_opt_float, default=None)
    p.add_argument("--min-improvement", type=_opt_float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-path", default=str(ROOT / "reports" / "promotion_report.json"))
    p.add_argument("--fail-on-reject", action="store_true")
    r = sub.add_parser("rollback")
    r.add_argument("--version", required=True)
    r.add_argument("--approved-dir", default=str(DEFAULT_APPROVED))
    r.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    ls = sub.add_parser("list")
    ls.add_argument("--approved-dir", default=str(DEFAULT_APPROVED))
    ls.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    a = ap.parse_args(argv)

    if a.cmd == "list":
        print(json.dumps(list_versions(a.approved_dir, a.registry_dir), indent=2))
        return 0
    if a.cmd == "rollback":
        print(json.dumps(rollback(a.version, a.approved_dir, a.registry_dir), indent=2))
        return 0
    d = promote_candidate(
        a.candidate_dir,
        a.approved_dir,
        a.registry_dir,
        tests_passed=a.tests_passed,
        min_pr_auc=a.min_val_pr_auc,
        min_recall=a.min_val_recall,
        min_improvement=a.min_improvement,
        dry_run=a.dry_run,
    )
    Path(a.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report_path).write_text(json.dumps(d, indent=2))
    _write_github_outputs(d)
    if d["promoted"]:
        print("PROMOTED")
    elif d.get("would_promote"):
        print("DRY RUN: all gates passed, candidate WOULD be promoted (nothing changed)")
    else:
        print("NOT PROMOTED (approved model preserved)")
    for c in d["checks"]:
        print(f"  [{'pass' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 1 if (a.fail_on_reject and not d["promoted"]) else 0


if __name__ == "__main__":
    sys.exit(main())
