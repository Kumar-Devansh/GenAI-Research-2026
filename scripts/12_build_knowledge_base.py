"""Backward-compatible entry point for the cited, incremental IMD PDF ingest."""
from knowledge_ingest import ingest_pdf

if __name__ == "__main__":
    print(f"Upserted {ingest_pdf()} cited document chunks into monsoon_grounded_v1.")
