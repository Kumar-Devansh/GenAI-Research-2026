"""FastAPI backend.  Run: uvicorn api:app --app-dir scripts --reload"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core import predict_next_day, search_knowledge
from agent import run_agent
from core import ROOT
from weather_sources import WeatherSourceError, open_meteo_forecast

app = FastAPI(title="Delhi Monsoon Weather Intelligence API")

class Q(BaseModel):
    question: str
@app.get("/")
def home():
    return {"status": "ok", "docs": "/docs", "endpoints": ["/health", "/forecast", "/weather/forecast", "/search?q=monsoon", "/ask"]}
@app.get("/health")
def health():
    return {"status": "ok", "archive_available": (ROOT / "data" / "delhi_weather_raw.csv").is_file()}
@app.get("/forecast")
def forecast():
    try: return predict_next_day()
    except Exception as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/weather/forecast")
def weather_forecast(city: str = "Delhi"):
    if city.lower() != "delhi": raise HTTPException(status_code=422, detail="Only Delhi coordinates are configured.")
    try: return open_meteo_forecast()
    except WeatherSourceError as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.get("/search")
def search(q: str):
    try: return {"results": search_knowledge(q)}
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc

@app.post("/ask")
def ask(body: Q):
    if not body.question.strip(): raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try: return {"answer": run_agent(body.question)}
    except Exception as exc: raise HTTPException(status_code=503, detail=f"Assistant unavailable: {exc}") from exc
