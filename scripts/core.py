"""Shared inference helpers. Saved-model output is a historical-data estimate."""
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
        with (ROOT / "models" / "meta.json").open(encoding="utf-8") as stream:
            _cache["meta"] = json.load(stream)
    return _cache

def predict_next_day() -> dict:
    """Estimate next-day mean temperature from the archive's last observed row."""
    c = _load(); meta = c["meta"]
    df = pd.read_csv(ROOT / "data" / "delhi_weather_raw.csv")
    dcol = next(x for x in df.columns if "date" in x.lower() or "time" in x.lower())
    df[dcol] = pd.to_datetime(df[dcol]); df = df.sort_values(dcol).set_index(dcol)
    f = df.select_dtypes("number").ffill().dropna()
    f["doy_sin"] = np.sin(2 * np.pi * f.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f.index.dayofyear / 365.25)
    missing = set(meta["cols"]) - set(f.columns)
    if missing:
        raise ValueError(f"Archive is missing model input columns: {', '.join(sorted(missing))}")
    f = f[meta["cols"]].tail(meta["window"])
    if len(f) != meta["window"]:
        raise ValueError("Not enough valid historical rows to form the model input window.")
    if not f.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("The final model window contains date gaps; rebuild a consecutive daily archive before inference.")
    x = c["sc"]["sx"].transform(f)[None]
    p = float(c["sc"]["sy"].inverse_transform(c["model"].predict(x, verbose=0))[0][0])
    nxt = (f.index[-1] + pd.Timedelta(days=1)).date().isoformat()
    return {"forecast_date": nxt, "predicted_mean_temp_c": round(p, 2), "model": meta["kind"],
            "source": "local historical archive", "product": "research-model estimate",
            "latest_input_date": f.index[-1].date().isoformat(), "is_live_forecast": False}

def search_knowledge(query: str, k: int = 3) -> str:
    """Search the monsoon knowledge base and return relevant passages."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if "emb" not in _cache:
        _cache["emb"] = SentenceTransformer("all-MiniLM-L6-v2")
    if "col" not in _cache:
        client = chromadb.PersistentClient(path=str(ROOT / "vectordb"))
        try:
            _cache["col"] = client.get_collection("monsoon_grounded_v1")
        except Exception as exc:
            raise FileNotFoundError("No source-grounded knowledge collection. Run scripts/knowledge_ingest.py first.") from exc
    r = _cache["col"].query(query_embeddings=_cache["emb"].encode([query], normalize_embeddings=True).tolist(),
                             n_results=max(1, min(10, int(k))))
    docs, metas = (r.get("documents") or [[]])[0], (r.get("metadatas") or [[]])[0]
    if not docs:
        return "No matching source passages were retrieved."
    return "\n---\n".join(f"[{m.get('source_title', 'Untitled source')} | page {m.get('page', '?')} | "
                             f"published {m.get('published', 'date unavailable')} | {m.get('source_url', 'URL unavailable')}] {d}"
                             for d, m in zip(docs, metas))
