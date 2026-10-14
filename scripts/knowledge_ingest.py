"""Non-destructive PDF/text ingestion with page-aware source metadata."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMD_REPORT = "https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf"
ALLOWED_HOSTS = {"mausam.imd.gov.in", "imd.gov.in", "internal.imd.gov.in", "moes.gov.in"}


def chunk_pages(pages: list[tuple[int, str]], size: int = 1000, overlap: int = 150) -> list[dict]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0.")
    out = []
    for page, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                out.append({"page": int(page), "text": chunk})
            if end == len(text):
                break
            start = end - overlap
    return out


def download_official_pdf(url: str, directory: Path, timeout: tuple[int, int] = (5, 45)) -> Path:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    host = (urlparse(url).hostname or "").lower()
    if urlparse(url).scheme != "https" or not any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise ValueError("Only HTTPS URLs on configured IMD/MoES source domains are allowed.")
    directory.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.4,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))))
    try:
        response = session.get(url, timeout=timeout, headers={"User-Agent": "WeatherIntelligenceResearch/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download official source: {exc}") from exc
    final_host = (urlparse(response.url).hostname or "").lower()
    if not any(final_host == h or final_host.endswith("." + h) for h in ALLOWED_HOSTS):
        raise ValueError("The source redirected outside the allowed IMD/MoES domains.")
    if len(response.content) > 100 * 1024 * 1024:
        raise ValueError("PDF exceeds the 100 MiB ingestion limit.")
    if not response.content.startswith(b"%PDF-"):
        raise ValueError("The official source URL did not return a PDF document.")
    digest = hashlib.sha256(response.content).hexdigest()
    destination = directory / f"{digest}.pdf"
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as tmp:
            tmp.write(response.content)
            temp_path = Path(tmp.name)
        temp_path.replace(destination)
    return destination


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf (requirements_app.txt) to ingest PDF sources.") from exc
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def ingest_pdf(url: str = DEFAULT_IMD_REPORT, title: str = "Observed Rainfall Features During the Southwest Monsoon 2024",
               published: str = "2024", root: Path = ROOT) -> int:
    """Download and incrementally upsert an official IMD PDF into a versioned collection."""
    doc_dir = root / "knowledge" / "documents"
    pdf = download_official_pdf(url, doc_dir)
    digest = pdf.stem
    pages = extract_pdf_pages(pdf)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("PDF text extraction produced no text; it may be scanned and require OCR.")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install requirements_app.txt for ChromaDB and sentence-transformers.") from exc
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(root / "vectordb"))
    collection = client.get_or_create_collection("monsoon_grounded_v1")
    ids = [f"{digest}_{i}" for i in range(len(chunks))]
    metadata = [{"source_title": title, "source_url": url, "published": published,
                 "page": c["page"], "sha256": digest, "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
                for c in chunks]
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True).tolist()
    # Replace this source's prior revision only after a valid replacement has
    # been downloaded, extracted, chunked and embedded successfully.
    collection.delete(where={"source_url": url})
    collection.upsert(ids=ids, documents=[c["text"] for c in chunks], metadatas=metadata, embeddings=embeddings)
    manifest = root / "knowledge" / "manifest.jsonl"
    row = {"source_title": title, "source_url": url, "published": published, "sha256": digest,
           "local_path": str(pdf.relative_to(root)), "pages": len(pages), "chunks": len(chunks),
           "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
    existing = []
    if manifest.exists():
        existing = [line for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line).get("source_url") != url]
    existing.append(json.dumps(row, ensure_ascii=False))
    manifest.write_text("\n".join(existing) + "\n", encoding="utf-8")
    return len(chunks)


if __name__ == "__main__":
    print(f"Upserted {ingest_pdf()} cited document chunks into monsoon_grounded_v1.")
