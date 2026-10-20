# Executed validation

Restoration date: 20 October 2026. Linux / Python 3.12.3. The evidence describes local execution, not a deployed Streamlit Community Cloud app.

## Automated tests

`python -m pytest -q`: **15 passed**.

The suite verifies:

- Portable loading independent of working directory; actionable missing-file errors.
- Schema/range validation; binary labels; duplicate keys; orphan references; contiguous cart positions.
- Temporal filters retaining all observed article lines of each selected basket, including an empty selection.
- Distinct-order activity counts and explicit article-count KPIs.
- Hand-calculated pair count, support, directional confidence and lift with baskets outside the candidate set in the denominator.
- Disjoint history/training/test order IDs and increasing within-customer sequences.
- Training/test label changes leaving model features unchanged; historical fallback for unseen IDs.
- Both models and the prevalence baseline actually fitted; metric ranges, baseline identities, deterministic repeated fits, confusion totals and unknown categorical values.
- Explicit failure on insufficient history.
- Synthetic and real Streamlit page navigation; synthetic model execution, model inspection, search, ranking and empty-filter state through Streamlit AppTest.
- Entire-basket exclusion when a target is missing, with audit manifest and hashes.

`ruff check app.py nextbuy_core scripts tests`: passes. Original academic scripts are excluded from lint enforcement and are not imported by the restored application.

`python -m pip check`: no broken requirements.

## Fresh installation and public-only snapshot

A separate, empty Python 3.12 environment at `/tmp/nextbuy-clean-env` was created and successfully installed using the root `requirements.txt`. The resulting transitive environment is recorded in `requirements.lock.txt`. The analytical evaluation was executed in that environment and reproduced the local results.

A separate application snapshot containing only `app.py`, `nextbuy_core/`, `.streamlit/` and `data/demo/` was tested with that clean environment. All five pages and the synthetic fixed model evaluation passed. It contains no original CSVs or local real cohort, so this explicitly verifies the fresh-clone data path.

## Scientific execution

`python scripts/audit_data.py` read all five supplied source CSVs and wrote `source_audit.json`.

`python scripts/prepare_data.py --local --users 3000` streamed the 13,692,886-line source, saved the cleaned cohort and source hashes, and preserved the original files. The clean cohort has 202,199 article lines across 20,256 orders.

`python scripts/evaluate.py` fitted and evaluated all three classifiers on both real and generated data. `evaluation.json` records metrics, reports, counts and timings. Local measurements show roughly 270 MiB peak RSS for both sources evaluated sequentially in one process; this is not a hosted memory limit measurement.

`jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --inplace --ExecutePreprocessor.timeout=180` completed. All four code cells have execution counts and no error outputs. The original academic notebook and checkpoint were read during audit and retained as historical evidence; their outdated logic was not used to establish the restored results.

## Running application and visual checks

The real Streamlit server was started with:

```bash
.venv/bin/python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

`python scripts/browser_check.py` uses actual Chromium against this server. It visits every page, executes the real-cohort experiment, changes a held-out article index, searches product names and downloads the resulting CSV. A keyboard interaction also changes and restores an hourly filter. All pages are checked for Streamlit exception elements and browser page errors. Plotly chart counts are recorded in `browser-check.json`.

Screenshots in `docs/screenshots/` cover all pages plus desktop (1440 px), tablet (768 px) and mobile (390 px). There is no document-level horizontal overflow at the tested tablet/mobile widths. Screenshots were visually inspected for layout and rendered graphs. Mobile columns stack; wide data tables and complex chart labels still benefit from a larger display. This is a bounded visual check, not an exhaustive accessibility or device certification.

## Issues found and corrected during verification

The first AppTest run identified an invalid Series reset-index argument on the quality page; it was corrected and the suite rerun. A test initially targeted the wrong selectbox by position; it now selects the widget by label. Early browser captures were taken before Plotly completed rendering; screenshot capture now waits for the footer and chart nodes.

Sandbox restrictions initially blocked package downloads and local sockets for Streamlit, Chromium and the Jupyter kernel. Those commands were rerun with the required execution approval and completed. They are not unresolved application failures.

## Remaining boundaries

- No external deployment, remote Git operation, hosted cold-start/resource check or repository visibility change was performed.
- No claim of future-purchase prediction, causal promotion benefit, externally calibrated probabilities or statistically significant model superiority is made.
- Real-data and historical notebook redistribution rights require review before publication; the generated demo runs independently of them.
- Original academic files are retained for traceability, rather than silently relabelled as corrected results.
