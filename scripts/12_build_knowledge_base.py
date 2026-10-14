"""Part 02 - Step 3: Build RAG vector DB (ChromaDB) from .txt/.md files in knowledge/."""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge"
KB.mkdir(exist_ok=True)

if not list(KB.glob("*.*")):
    (KB / "monsoon_basics.txt").write_text(
        "The Indian Southwest Monsoon typically sets over Kerala around 1 June and covers Delhi by late June to early July.\n\n"
        "Monsoon withdrawal from northwest India usually begins around 17 September.\n\n"
        "Heat waves in Delhi occur when maximum temperature crosses 40 C and departs 4.5 C or more from normal.\n\n"
        "Heavy rainfall is 64.5 to 115.5 mm in 24 hours; very heavy is 115.6 to 204.4 mm.\n", encoding="utf-8")
    print("Created starter file. Add your own IMD reports/papers (.txt/.md) in knowledge/ and re-run.")

def chunks(text, size=600):
    out, cur = [], ""
    for p in text.split("\n\n"):
        if len(cur) + len(p) > size and cur:
            out.append(cur.strip()); cur = ""
        cur += p + "\n\n"
    if cur.strip(): out.append(cur.strip())
    return out

emb = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=str(ROOT / "vectordb"))
try: client.delete_collection("monsoon")
except Exception: pass
col = client.create_collection("monsoon")

docs, ids, metas = [], [], []
for f in list(KB.glob("*.txt")) + list(KB.glob("*.md")):
    for i, c in enumerate(chunks(f.read_text(encoding="utf-8"))):
        docs.append(c); ids.append(f"{f.stem}_{i}"); metas.append({"source": f.name})
col.add(documents=docs, ids=ids, metadatas=metas, embeddings=emb.encode(docs).tolist())
print(f"Indexed {len(docs)} chunks.")
