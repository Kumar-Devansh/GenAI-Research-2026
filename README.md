# 🌦️ Development of an Intelligent Multi-Modal Weather Intelligence and Climate Decision Support Platform Integrating Machine Learning, Deep Learning, Computer Vision, Remote Sensing, Retrieval-Augmented Generation (RAG), AI Agents, and Large Language Models (Part 01)

**MacroEdtech — Team GenAI Research 2026**
*Phase 02: Demonstration of Technical Skills*

> Part 01 of an ongoing research project building an intelligent, multi-modal Weather Intelligence and Climate Decision Support Platform for the Indian Southwest Monsoon — integrating Machine Learning, Deep Learning, Computer Vision, and (in Part 02) Retrieval-Augmented Generation (RAG), AI Agents, and LLMs.

**Author:** Kanishka
**Mentor:** Sagar Sakalley, Founder & Director, MacroEdtech

---

## 📌 Overview

This repository contains the complete pipeline for **Part 01**: an end-to-end, open-source weather prediction system for the **Delhi region**, covering:

- Historical meteorological data acquisition (Open-Meteo Historical Weather API)
- Exploratory Data Analysis (EDA)
- Feature engineering (lag features, rolling statistics, cyclical encodings)
- **10 models across 4 paradigms**, trained and comparatively evaluated for next-day mean temperature prediction:
  - **5 Machine Learning regressors:** Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, XGBoost
  - **3 Statistical time-series models:** ARIMA, SARIMA, Prophet
  - **2 Deep Learning sequence models:** LSTM, GRU
  - **1 Computer Vision proof-of-concept:** CNN for sky/cloud condition classification

## 🏆 Key Results

| Model Family | Best R² | Notes |
|---|---|---|
| Multivariate ML (Linear Regression) | **0.947** | Uses humidity, pressure, wind speed, lag features |
| Deep Learning (GRU) | **0.921** | Univariate — temperature sequence only |
| Statistical Time-Series (Prophet) | **0.207** | Univariate — temperature sequence only |
| CNN (Sky Image Classification) | **97.8% accuracy** | Synthetic sky image dataset (3 classes) |

**Key finding:** Both richer input features *and* richer (non-linear) model architectures independently improve forecasting accuracy — LSTM/GRU close most of the gap to multivariate ML despite using no auxiliary features at all.

---

## 📂 Project Structure

```
weather_project/
├── data/
│   ├── sky_images/              # CNN training data (clear_sky, partly_cloudy, heavy_clouds)
│   ├── delhi_weather_raw.csv    # Raw historical weather data
│   └── delhi_weather_features.csv  # Engineered feature set
├── models/                      # Saved/trained model artifacts
├── outputs/                     # Plots, dashboards, and result CSVs
│   ├── cnn_classification_results.png
│   ├── correlation_heatmap.png
│   ├── deep_learning_comparison.csv
│   ├── deep_learning_results.png
│   ├── eda_plots.png
│   ├── model_comparison.csv
│   ├── model_results.png
│   ├── part01_summary_dashboard.png
│   ├── timeseries_model_comparison.csv
│   └── timeseries_results.png
└── scripts/
    ├── 01_data_collection.py
    ├── 01b_generate_sample_data.py
    ├── 02_eda.py
    ├── 03_feature_engineering.py
    ├── 04_train_ml_models.py
    ├── 05_inspect_model.py
    ├── 06_timeseries_models.py
    ├── 07_summary_dashboard.py
    ├── 08_deep_learning_models.py
    ├── 09a_generate_sky_images.py
    └── 09b_cnn_classification.py
```

## ⚙️ Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn, XGBoost
- **Time-Series:** statsmodels (ARIMA/SARIMAX), Prophet
- **Deep Learning:** TensorFlow / Keras (LSTM, GRU, CNN)
- **Data:** pandas, NumPy, Open-Meteo Historical Weather API
- **Visualization:** Matplotlib / Seaborn

## 🚀 Getting Started

### Prerequisites
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline
Scripts are numbered in execution order:
```bash
python scripts/01_data_collection.py        # or 01b_generate_sample_data.py (offline fallback)
python scripts/02_eda.py
python scripts/03_feature_engineering.py
python scripts/04_train_ml_models.py
python scripts/06_timeseries_models.py
python scripts/08_deep_learning_models.py
python scripts/09a_generate_sky_images.py
python scripts/09b_cnn_classification.py
python scripts/07_summary_dashboard.py
```

Results (plots, CSVs, dashboard) are saved to the `outputs/` folder.

## 📊 Dataset

- **Source:** Open-Meteo Historical Weather API (free, open-source, no license required)
- **Location:** Delhi NCR (~28.61°N, 77.21°E)
- **Period:** 2019–2024 (2,192 daily records)
- **Variables:** Max/min/mean temperature, precipitation, humidity, pressure, wind speed, cloud cover

A synthetic data generator (`01b_generate_sample_data.py`) is included as a fallback for offline environments.

## ⚠️ Limitations

- Single-location study (Delhi only)
- SARIMA uses a 7-day seasonal period for tractability, not the full annual cycle
- Time-series and deep learning models evaluated only in univariate form
- CNN trained on procedurally generated synthetic sky images, not real satellite imagery
- Scope limited to next-day mean temperature prediction

## 🔭 Roadmap — Part 02

- Multivariate LSTM/GRU with auxiliary features + real Sentinel-2/MODIS satellite imagery
- Advanced CV: YOLOv8 object detection, U-Net/SAM segmentation
- Generative AI assistant: RAG, vector databases (FAISS/ChromaDB), LangChain/LangGraph agents, MCP tool integration, open-weight LLMs
- Rainfall and multi-variable forecasting
- Deployment via FastAPI + Streamlit/Gradio, containerized with Docker

## 📄 Citation / Paper

Full methodology, literature review, and results are documented in the accompanying research paper (`paper/` or root `.tex` file — Part 01).


---
*MacroEdtech — Team GenAI Research 2026*
