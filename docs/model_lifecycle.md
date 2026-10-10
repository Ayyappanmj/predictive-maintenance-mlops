# Model lifecycle: versioning, promotion, rollback, delivery

## Versions
Each training run writes `model.joblib`, `metadata.json` and `reference.csv`. The version is a UTC
timestamp (`YYYYMMDD-HHMMSS`). `metadata.json` holds the seed, library versions, data hash,
validation and test metrics, decision threshold, and the SHA-256 of `model.joblib`.

## Layout
| Path | Purpose |
|---|---|
| `models/` | the approved model served by the API and dashboard |
| `model_registry/candidate/` | the candidate trained by the workflow |
| `model_registry/archive/<version>/` | every previously approved model (rollback source) |
| `model_registry/promotion_log.jsonl` | every decision, including rejection reasons |

## Promotion gates (all must pass; validation data only, never the test set)
1. Artifact integrity: all files present and the model hash matches metadata.
2. Automated tests passed (`--tests-passed`, default false: fails closed).
3. Input schema equals the API/config schema.
4. API compatibility: the candidate loads and predicts through the API code path.
5. Validation PR-AUC >= `promotion.min_val_pr_auc`.
6. Validation recall >= `promotion.min_val_recall`.
7. Validation PR-AUC >= approved model's + `promotion.min_improvement`.

If any gate fails the approved model is not touched and reasons are logged.
Limitations: the comparison trusts `metadata.json` produced by the workflow. Approved and
candidate validation scores are comparable only while data and seed (hence the split) are the
same. The thresholds in the config are initial settings; review them against your own results.

## Rollback
```bash
python -m src.promote_model list
python -m src.promote_model rollback --version 20260101-030000
```
Rollback verifies the archived model's hash, archives the model being replaced, then installs it.
On Render, the restored model is only picked up after the service restarts, and a restart that
runs `fetch_model.py` would fetch the latest release again; to pin an older model, publish it.

## How an approved model reaches the deployed app
1. `retrain.yml` promotes a candidate inside the workflow runner (its disk is temporary).
2. If publishing is enabled (manual run with `publish: true`, or repository variable
   `PUBLISH_MODEL=true`), the `publish` job uploads `model-bundle.tar.gz` to GitHub Releases:
   `model-<version>` (history) and `model-latest` (rolling pointer).
3. A deployment runs `python scripts/fetch_model.py --repo OWNER/REPO` on start (Phase 10) which
   downloads the bundle, verifies the hash and installs it into `models/`.

**Not implemented or tested:** automatic redeploys. After publishing, the deployed service picks up
the new model only on its next start or deploy. Render's free filesystem is temporary, so the
model is re-fetched on every start. Public repositories only.

## GitHub Actions notes
- Scheduled workflows run from the default branch, in UTC, and can be delayed. GitHub disables
  schedules in public repositories after 60 days without repository activity.
- No workflow commits or pushes, and none triggers on releases, so there are no trigger loops.
- Permissions are read-only except the `publish` job (`contents: write`). No secrets are needed.
- Third-party actions use version tags; Dependabot proposes updates. Pin SHAs for stricter security.

## Manual retraining (no GitHub needed)
```bash
python scripts/download_data.py
python -m src.train --model-dir model_registry/candidate --report-dir reports/candidate
pytest -q
python -m src.promote_model promote --tests-passed true          # add --dry-run to preview
```
On GitHub: Actions > "Scheduled retraining" > Run workflow.
