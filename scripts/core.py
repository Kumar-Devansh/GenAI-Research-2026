"""Shared helpers: forecast + knowledge search."""
import json
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parent.parent
_cache = {}

def _load():
    if not _cache:
        from tensorflow.keras.models import load_model
        _cache["model"] = load_model(ROOT / "models" / "best_multivariate.keras")
        _cache["sc"] = joblib.load(ROOT / "models" / "scalers.joblib")
        _cache["meta"] = json.load(open(ROOT / "models" / "meta.json"))
    return _cache

def predict_next_day() -> dict:
    """Forecast next-day mean temperature (C) for Delhi from the latest data."""
    c = _load(); meta = c["meta"]
    df = pd.read_csv(ROOT / "data" / "delhi_weather_raw.csv")
    dcol = next(x for x in df.columns if "date" in x.lower() or "time" in x.lower())
    df[dcol] = pd.to_datetime(df[dcol]); df = df.sort_values(dcol).set_index(dcol)
    f = df.select_dtypes("number").interpolate().dropna()
    f["doy_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)
    f = f[meta["cols"]].tail(meta["window"])
    x = c["sc"]["sx"].transform(f)[None]
    p = float(c["sc"]["sy"].inverse_transform(c["model"].predict(x, verbose=0))[0][0])
    nxt = (f.index[-1] + pd.Timedelta(days=1)).date().isoformat()
    return {"forecast_date": nxt, "predicted_mean_temp_c": round(p, 2), "model": meta["kind"]}

def search_knowledge(query: str, k: int = 3) -> str:
    """Search the monsoon knowledge base and return relevant passages."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    if "emb" not in _cache:
        _cache["emb"] = SentenceTransformer("all-MiniLM-L6-v2")
        _cache["col"] = chromadb.PersistentClient(path=str(ROOT / "vectordb")).get_collection("monsoon")
        r = _cache["col"].query(query_embeddings=_cache["emb"].encode([query]).tolist(), n_results=int(k))

    return "\n---\n".join(f"[{m['source']}] {d}" for d, m in zip(r["documents"][0], r["metadatas"][0]))
