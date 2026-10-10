# Portfolio and interview notes

## 60-second pitch
I built an end-to-end predictive maintenance system on a public synthetic dataset. It trains and
compares a baseline and a Random Forest with leakage-safe preprocessing and validation-based model
and threshold selection, serves predictions through a validated FastAPI service and a Streamlit
dashboard, monitors incoming data for drift, and retrains through GitHub Actions. A new model only
replaces the old one if it passes explicit gates, and the previous model is archived for rollback.
I was careful to document what it does not do: the data is synthetic, the scores are uncalibrated,
and it does not predict time to failure.

## Decisions you can defend
| Decision | Why |
|---|---|
| Removed `TWF/HDF/PWF/OSF/RNF` | They define "Machine failure", so using them is target leakage |
| Preprocessing inside a sklearn `Pipeline` | Guarantees it is fitted on training data only |
| PR-AUC for selection, recall floor in gates | Failures are about 3% of rows; accuracy would be misleading |
| Threshold chosen on validation | Avoids tuning on the test set; test used once |
| Class weights instead of SMOTE | Simple, no extra dependency, no synthetic rows |
| SciPy drift tests, Evidently optional | Lightweight, testable offline; Evidently's API changes across versions |
| Gates read validation metrics only | Prevents peeking at the test set when promoting |
| GitHub Releases for model delivery | Render's free disk is temporary; releases are free and hash-checked |
| No auto redeploy claim | It is not implemented or tested, so the docs say so |

## Questions to expect
- *Why not accuracy?* With about 3% positives, predicting "no failure" scores ~97%.
- *Data drift vs concept drift?* Inputs changing vs the input-to-label relationship changing; the first does not prove accuracy fell, and measuring the second needs labels.
- *Are the probabilities calibrated?* No: class weights shift them. Next step: calibration on validation data.
- *How would you improve the promotion comparison?* Evaluate both models on one fixed holdout set and sign the metadata.
- *What would change with real data?* Timestamped features, group-aware splits per machine, label delay handling, calibration, and real monitoring storage.

## Resume bullet templates (fill in only numbers you measured)
- Built an end-to-end predictive-maintenance pipeline (scikit-learn, FastAPI, Streamlit) with leakage-safe preprocessing; selected a Random Forest on validation PR-AUC of **[your value]** and evaluated once on a held-out test set (recall **[value]**).
- Implemented data-drift monitoring (KS/chi-square/PSI) with configurable alerts and reproducible simulation; covered by **[N]** automated tests.
- Designed GitHub Actions CI and scheduled retraining with gated model promotion, archive and rollback, so a failing candidate never replaces the approved model.
- Documented free-tier Render deployment trade-offs (cold starts, ephemeral storage) and delivered models through hash-verified GitHub Releases.

## Demo script (3 minutes)
1. Dashboard Predict tab: enter a reading, show risk category and the disclaimer.
2. Model performance tab: explain PR-AUC, recall, and the confusion matrix honestly.
3. Monitoring tab: run the simulated stream and show alerts appearing after the injected shift.
4. GitHub: show CI and a retraining run's promotion table, then a rejected candidate's reasons.
