# Streamlit Community Cloud preparation

Prepared on Python 3.12.3 with a successful installation in a separate empty virtual environment. No external deployment, commit, push or repository visibility change was performed.

## Files to review and publish after approval

- `app.py`, `nextbuy_core/`, `requirements.txt`, `requirements.lock.txt`, `.streamlit/config.toml`.
- `data/demo/*.csv` and `manifest.json`: generated demonstration data, about 0.6 MB.
- README, reproducible scripts, tests and relevant documentation.
- Historical academic artefacts require the authorship/data review in `DATA.md`; they are not needed to run the hosted demo.

Do not publish `.venv/`, `data/local/`, raw `nextbuy/datasets/*.csv`, secrets or browser test downloads. `.gitignore` excludes them. The original 233 MiB line CSV exceeds ordinary GitHub per-file limits and is unnecessary for the public demo.

## Deployment after user authorisation

1. Review the working files and academic publication rights. Commit and push only after explicit authorisation.
2. In Streamlit Community Cloud, create an app from `aitbenamaraabdelkader27/Nextbuy-project` and the branch containing the reviewed implementation.
3. Set the main file to `app.py` and select Python **3.12** under Advanced settings. `.python-version` is a local version hint, not a substitute for that setting.
4. No secrets are required. The root `.streamlit/config.toml` configures theme and headless mode. No system package, GPU, database or paid API is needed.
5. Verify that the visible source is **Synthetic demo**, visit all five pages and execute the fixed model evaluation. Inspect logs for installation and memory errors. Do not label the public synthetic metrics as real Instacart performance.
6. Add the authorised hosted URL to the README and portfolio only once it exists.

Direct dependencies are pinned. `requirements.lock.txt` records the clean tested transitive environment and may be used locally with `pip install -r requirements.lock.txt`; the cloud dependency file is `requirements.txt`. Update both together after verification. The deployed runtime does not require Jupyter, pytest, Ruff, Playwright or Matplotlib.

## Resource scope

Data loading is bounded to a compact cohort. The raw large source is never loaded by the dashboard. Page navigation renders only the selected page. Data cache and model resource cache each have two entries. Forest training uses one CPU worker, fixed small trees, and runs on explicit request. There are no binary model files or runtime downloads. A local evaluation of both sources in a single process used around 272 MiB peak RSS; cloud cold starts and hosted memory are still untested. Streamlit hosting is the intended 0 €/month option; availability and free-tier quotas are platform-controlled.

Official references, consulted during restoration: [deployment and Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency management](https://docs.streamlit.io/deploy/concepts/dependencies), [app management and resource troubleshooting](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).
