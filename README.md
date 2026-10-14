# Intelligent Multi-Modal Weather Intelligence and Climate Decision Support Platform — Part 01

**MacroEdtech — Team GenAI Research 2026** · Delhi, India · Historical weather study

Part 01 collects daily weather observations, explores and engineers predictors, evaluates forecasting models, and trains a synthetic-image sky classifier. It contains **10 forecasting models** (5 conventional ML, 3 statistical time-series, 2 sequence models) plus a separate CNN image-classification proof of concept. Part 02/RAG/API scripts are present in `scripts/` but are outside the Part 01 run sequence.

## Project layout

```text
data/       Raw/engineered weather CSVs and synthetic sky images
models/     Trained model artifacts
outputs/    Metrics CSVs, plots, and summary dashboard
scripts/    Numbered Part 01 pipeline, shared helpers, and optional Part 02 scripts
knowledge/  Part 02 knowledge-base source files
vectordb/   Part 02 ChromaDB data
```

Weather CSVs and trained artifacts are local inputs/outputs and may be regenerated. CNN image data is generated procedurally and is not satellite imagery. The repository's existing output files are from previous runs; headline scores in earlier project notes should be treated as historical and unverified until the steps below are rerun on the same data and environment.

## Requirements and setup

Use Python 3.10–3.12. TensorFlow and Prophet availability varies by platform and Python version; use a fresh virtual environment.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Part 01

Run commands from the repository root. Scripts resolve project files from their own location, so they also work when launched from another working directory. The API collection uses Open-Meteo and needs network access.

```bash
python scripts/01_data_collection.py
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_train_ml_models.py
python scripts/06_timeseries_models.py
python scripts/08_deep_learning_models.py
python scripts/09a_generate_sky_images.py
python scripts/09b_cnn_classification.py
python scripts/07_summary_dashboard.py
```

For offline demonstrations only, explicitly run `python scripts/01b_generate_sample_data.py` instead of data collection. This writes clearly synthetic observations to `data/delhi_weather_raw.csv` and overwrites that file. It is not real weather and must not be reported as observed data. The image generator similarly produces synthetic sky patterns.

Optional inspection: `python scripts/05_inspect_model.py` prints details for the saved Gradient Boosting model; run it after step 04.

Outputs include `model_comparison.csv`, `timeseries_model_comparison.csv`, `deep_learning_comparison.csv`, `cnn_classification_comparison.csv`, model artifacts, plots, and `part01_summary_dashboard.png` under `outputs/` and `models/`.

## Evaluation and interpretation

- The forecasting scripts use chronological splits with the same 80% date boundary and next-day temperature target. ML features describe the observation day; their target is the following day's mean temperature.
- The ML scaler is fitted on training rows only. LSTM/GRU input scaling is also fitted only on the training period. Sequence test windows may use earlier observed values, as they would in a rolling forecast.
- ARIMA, SARIMA, and Prophet are univariate; engineered-feature ML and LSTM/GRU evaluations have different inputs. Interpret the scores with that distinction in mind. The reported holdout is one fixed period, not a rolling-origin cross-validation study.
- Forecast results use MAE, RMSE, and R². CNN accuracy is a separate task and must not be compared with forecast R². CNN epoch selection uses a validation split; its held-out test split is evaluated after training. Since the images are synthetic, accuracy measures fit to the generator, not real sky or satellite classification capability.
- The dashboard reports only metrics present in result CSVs. It does not insert prior or approximate scores when a model family has not been run.

## Part 02 (optional)

`scripts/10_multivariate_lstm_gru.py`, `11_yolo_sky_classifier.py`, `12_build_knowledge_base.py`, plus `core.py`, `api.py`, `agent.py`, and `app.py` are additional Part 02 components. Install their extra packages from `requirements_part2.txt` after the Part 01 requirements. The forecast/API and RAG functions require their saved models/vector database and any associated services; Part 01 does not build all of those artifacts.

Install the UI dependencies with `python -m pip install -r requirements_app.txt`, then launch the Streamlit research interface from the repository root with `streamlit run scripts/app.py`. It includes overview, weather, monsoon, satellite-demo, model-results, assistant, and knowledge-search views. Local charts use the historical CSV; the saved-model estimate is explicitly dated to that archive, and provider forecasts are separately attributed to Open-Meteo. RAG uses the grounded, page-cited collection described below. Satellite samples remain labeled synthetic, and unavailable integrations show an empty or explicit error state. The UI is limited to Delhi. Install `requirements_part2.txt` separately only when using YOLO and the optional API integrations.

## Part 02 research upgrades

The provider and research additions deliberately distinguish three kinds of information:

- **Provider forecast/current model conditions:** `/weather/forecast` and the overview's “Fetch current model conditions” action query Open-Meteo Forecast API for Delhi and display provider time, fetch time, units, and attribution. Open-Meteo documents its current conditions as weather-model data; these are not local station observations. Review the [Forecast API documentation](https://open-meteo.com/en/docs).
- **Official IMD observations:** optional and credential-gated. Set `IMD_API_AUTH_HEADER`, `IMD_API_AUTH_VALUE`, and `IMD_STATION_ID` according to the exact auth scheme and station ID from your [IMD API account](https://api.imd.gov.in/public/api_reference.html). The public reference documents the Current Weather endpoint and observation fields, but does not specify the authentication scheme; the app therefore does not guess it.
- **Saved research-model estimate:** `/forecast` and the original model button still run the saved LSTM/GRU against the local archive. The result is explicitly marked `is_live_forecast: false` and dated relative to the last archive row. It is not a same-day operational forecast.

The historical CSV contains Open-Meteo archive values through 2024; it is not an independently verified station record. Current provider values are not mixed into that training archive automatically. Do not interpret historical archive estimates as IMD observations.

### Grounded IMD document retrieval

Install `requirements_app.txt` and run:

```bash
python scripts/knowledge_ingest.py
```

This downloads the official [IMD Southwest Monsoon 2024 report](https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf), verifies HTTPS against an IMD/MoES host allowlist and PDF signature, hashes the document, extracts page-aware chunks, and upserts them into `vectordb/` collection `monsoon_grounded_v1`. Retrieval results include title, publication year, source URL, and page number. Re-running the same source is idempotent; a successfully processed update replaces the chunks for that URL. Existing `monsoon` data is left untouched and is not treated as verified evidence. Scanned/image-only PDFs fail with a clear extraction error; OCR is not silently applied. The first RAG use may download the Sentence Transformers embedding model.

### Rainfall experiment

Run after installing Part 01 dependencies:

```bash
python scripts/rainfall_models.py
```

The experiment predicts next-day rainfall occurrence (`> 0 mm`) and amount using only same-day/lagged predictors. It partitions chronologically into 70% train, 15% validation, and 15% test; the F1 threshold is selected on validation only. It reports precision, recall, F1, average precision (PR-AUC), confusion matrix, amount MAE/RMSE, and a training-period prior/median baseline in `outputs/rainfall_experiment_metrics.json` and test predictions in `outputs/rainfall_experiment_predictions.csv`. Metrics are not written into the existing Part 01 comparison files. The label source is the historical Open-Meteo archive, the wet/dry threshold is exact zero, and the resulting workflow is experimental—not an operational rainfall forecast.

### Sentinel-2 catalog and segmentation workflow

Metadata-only Sentinel-2 Level-2A search (no imagery is downloaded):

```bash
python scripts/satellite_data.py --bbox 77.0 28.4 77.4 28.9 --start 2024-07-01 --end 2024-07-31 --max-cloud 20 --limit 5
```

Searches use the current [Copernicus Data Space STAC API](https://documentation.dataspace.copernicus.eu/APIs/STAC.html) and report product IDs, acquisition time, cloud cover, and asset names. Keep the area/time window small. Authenticated asset download and band-stack generation are not automated in this phase. `scripts/satellite_data.py` provides `validate_paired_tiles()` and `train_unet()`: callers must supply separate, disjoint train/validation/test image-mask pairs and a loader that returns normalized reflectance and binary, task-specific masks. The helper reports held-out Dice/IoU and saves the model; there are no bundled scenes or labels, so no model is trained or accuracy claimed by the metadata search. Synthetic sky imagery and the existing image classifier remain a separate experiment.

### API and Docker Compose

Local services require Python 3.11 recommended for TensorFlow compatibility, and the repo's `data/` and `models/` artifacts. Copy `.env.example` to `.env` and fill values only for configured optional integrations (never commit `.env`):

```dotenv
IMD_API_AUTH_HEADER=
IMD_API_AUTH_VALUE=
IMD_STATION_ID=
OLLAMA_HOST=http://host.docker.internal:11434
```

Start the API and UI locally in Docker:

```bash
docker compose up --build
```

Open `http://localhost:8501` (UI), `http://localhost:8000/docs` (API), and `http://localhost:8000/health` (health check). Data and model mounts are read-only; Chroma and source documents use local persistent mounts. `requirements_runtime.txt` omits YOLO, XGBoost, Prophet, OpenCV, and training-only libraries; it includes CPU TensorFlow for the saved model and RAG dependencies, so the image remains substantial. Ollama is optional; run it on the host and set `OLLAMA_HOST` if using the assistant. Stop with `docker compose down`. This is local development configuration only; no cloud resources are provisioned or exposed publicly.

### Tests and limitations

Run the offline tests and syntax checks:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/weather_sources.py scripts/rainfall_models.py scripts/knowledge_ingest.py scripts/satellite_data.py scripts/core.py scripts/api.py scripts/app.py
```

External provider access, the IMD credential-gated endpoint, RAG model download, deep model training, authenticated image assets, and U-Net training depend on network access, optional packages/credentials, and real labeled data. The repository currently has no satellite samples or masks. Copernicus catalog search is metadata-only and its endpoint can evolve; asset-level data download requires following current CDSE authentication/usage guidance. No SAM integration was added because it would require additional weights and compute without a validated labeled task. No public AWS deployment is configured.

## Troubleshooting

- **Missing weather CSV:** run step 01 (real API data) or explicitly choose step 01b (synthetic demonstration data).
- **Open-Meteo request failure:** check network access and retry. A failed request leaves the existing CSV untouched.
- **Missing or malformed columns:** keep the Open-Meteo column mapping and rerun data collection, then rerun feature engineering before training.
- **Dashboard says results are missing:** run the corresponding training script first. CNN and sequence-model cards remain marked unavailable until their result files exist.
- **TensorFlow/Prophet installation fails:** use Python 3.10–3.12 in a clean environment and consult the package installation instructions for the active OS/CPU.
- **CPU training is slow:** deep learning and SARIMA steps can take longer than the tabular ML scripts. They keep their existing model families and fixed configurations; no speed/accuracy claims are made without a rerun.

## Limitations

Single location (Delhi); daily historical data; a single chronological holdout rather than rolling-origin validation; limited model tuning; SARIMA's seasonal period is 7 days; and synthetic CNN imagery. The Part 01 study forecasts next-day mean temperature. The accompanying paper mentioned in earlier README text is not included in this repository snapshot.
