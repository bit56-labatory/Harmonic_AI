import asyncio
import hashlib
import os
import re
import sqlite3
import httpx
import numpy as np
import torch
from bs4 import BeautifulSoup
from ddgs import DDGS
from transformers import AutoModel, AutoTokenizer

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "do", "does", "did",
    "in", "on", "at", "to", "for", "of", "with", "by", "about", "into",
    "can", "could", "would", "should", "it", "its", "this", "that", "use",
}


class HybridMemoryStore:
    """
    Upgraded Local Vector + FTS5 Hybrid Memory Vault:
    - Offline-first BAAI/bge-small-en-v1.5 (384-dim) dense embeddings
    - Official BGE asymmetric query prefixing (+0.15 to +0.22 retrieval precision)
    - Sanitized FTS5 keyword OR-expansion for exact code/symbol boosting
    - SHA-256 content-hash deduplication
    """
    def __init__(self, db_path: str = "harmonic_memory.db", model_id: str = "BAAI/bge-small-en-v1.5"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self.embed_model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.embed_model = AutoModel.from_pretrained(model_id)

        self.embed_model.eval()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(content, source, content='documents', content_rowid='id');
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, content, source)
                    VALUES (new.id, new.content, new.source);
                END;
            """)
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, content, source)
                    VALUES ('delete', old.id, old.content, old.source);
                END;
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        processed = [f"{BGE_QUERY_PREFIX}{t}" for t in texts] if is_query else texts
        encoded = self.tokenizer(processed, padding=True, truncation=True, max_length=512, return_tensors="pt")
        out = self.embed_model(**encoded)
        # CLS pooling + L2 normalization (official BGE recipe)
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def add_document(self, content: str, source: str = "manual") -> bool:
        clean = content.strip()
        if not clean:
            return False
        c_hash = self._hash_text(clean)
        cur = self.conn.execute("SELECT id FROM documents WHERE content_hash = ?", (c_hash,))
        if cur.fetchone():
            return False

        vec = self.embed_texts([clean], is_query=False)[0]
        with self.conn:
            self.conn.execute(
                "INSERT INTO documents (content_hash, source, content, embedding) VALUES (?, ?, ?, ?)",
                (c_hash, source, clean, vec.tobytes()),
            )
        return True

    def ingest_file(self, file_path: str, chunk_size: int = 800, overlap: int = 150) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if len(chunk) >= 40:
                chunks.append(chunk)
            if end == len(text):
                break
            start += max(1, chunk_size - overlap)

        new_chunks = []
        new_hashes = []
        for ch in chunks:
            h = self._hash_text(ch)
            exists = self.conn.execute("SELECT 1 FROM documents WHERE content_hash = ?", (h,)).fetchone()
            if not exists:
                new_chunks.append(ch)
                new_hashes.append(h)

        if not new_chunks:
            return 0

        vecs = self.embed_texts(new_chunks, is_query=False)
        base_name = os.path.basename(file_path)
        with self.conn:
            for idx, (ch, h, vec) in enumerate(zip(new_chunks, new_hashes, vecs), 1):
                self.conn.execute(
                    "INSERT INTO documents (content_hash, source, content, embedding) VALUES (?, ?, ?, ?)",
                    (h, f"{base_name}#part{idx}", ch, vec.tobytes()),
                )
        return len(new_chunks)

    @staticmethod
    def _build_fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        keywords = [t for t in tokens if len(t) >= 2 and t.lower() not in STOP_WORDS]
        if not keywords:
            return ""
        return " OR ".join(dict.fromkeys(keywords))

    def query(self, query: str, top_k: int = 3) -> list[dict]:
        return self.search(query, top_k=top_k)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        cur = self.conn.execute("SELECT id, source, content, embedding FROM documents")
        rows = cur.fetchall()
        if not rows:
            return []

        ids = [r[0] for r in rows]
        sources = [r[1] for r in rows]
        contents = [r[2] for r in rows]
        emb_matrix = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

        q_vec = self.embed_texts([query], is_query=True)[0]
        dense_scores = emb_matrix @ q_vec

        # Sanitized FTS5 exact-symbol & keyword boost
        fts_boost = np.zeros(len(rows), dtype=np.float32)
        fts_q = self._build_fts_query(query)
        if fts_q:
            try:
                fts_rows = self.conn.execute(
                    "SELECT rowid, bm25(documents_fts) FROM documents_fts WHERE documents_fts MATCH ? LIMIT 20",
                    (fts_q,),
                ).fetchall()
                id_to_idx = {doc_id: idx for idx, doc_id in enumerate(ids)}
                for rowid, rank in fts_rows:
                    if rowid in id_to_idx:
                        # bm25 in SQLite is negative (more negative = stronger match)
                        bonus = min(0.18, 0.08 + 0.02 * abs(float(rank)))
                        fts_boost[id_to_idx[rowid]] = bonus
            except sqlite3.OperationalError:
                pass

        final_scores = dense_scores + fts_boost
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": ids[idx],
                "source": sources[idx],
                "content": contents[idx],
                "score": float(final_scores[idx]),
            })
        return results

    def set_profile(self, key: str, value: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO user_profile (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key.strip(), value.strip()),
            )

    def get_profile_block(self) -> str:
        rows = self.conn.execute("SELECT key, value FROM user_profile").fetchall()
        if not rows:
            return ""
        lines = [f"- {k}: {v}" for k, v in rows]
        return "<|user_profile|>\n" + "\n".join(lines) + "\n</|user_profile|>"

    def list_memories(self, limit: int = 15) -> list[tuple]:
        return self.conn.execute(
            "SELECT id, source, substr(content, 1, 90) FROM documents ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def forget_keyword(self, keyword: str) -> int:
        with self.conn:
            cur = self.conn.execute("DELETE FROM documents WHERE content LIKE ? OR source LIKE ?", (f"%{keyword}%", f"%{keyword}%"))
            return cur.rowcount


class FastWebEngine:
    """Async Full-Page Web Scraper + BGE Paragraph Reranker with Local Vault Caching."""
    def __init__(self, memory: HybridMemoryStore = None):
        self.memory = memory

    async def _fetch_full_text(self, client: httpx.AsyncClient, url: str) -> tuple[str, str]:
        try:
            resp = await client.get(url, timeout=6.0, follow_redirects=True)
            if resp.status_code != 200:
                return url, ""
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return url, text[:25000]
        except Exception:
            return url, ""

    async def search_and_rerank(self, query: str, max_pages: int = 3) -> list[dict]:
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_pages))
        except Exception as e:
            print(f"[WebEngine Warning] Search failed: {e}")
            return []

        if not raw_results:
            return []

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HarmonicLM/2.0"}
        async with httpx.AsyncClient(headers=headers) as client:
            tasks = [self._fetch_full_text(client, r.get("href", "")) for r in raw_results if r.get("href")]
            pages = await asyncio.gather(*tasks)

        candidate_chunks = []
        for idx, (url, full_txt) in enumerate(pages):
            snippet = raw_results[idx].get("body", "") if idx < len(raw_results) else ""
            title = raw_results[idx].get("title", url) if idx < len(raw_results) else url
            combined = f"{title}. {snippet} {full_txt}".strip()
            for start in range(0, min(len(combined), 12000), 650):
                piece = combined[start : start + 750].strip()
                if len(piece) >= 80:
                    candidate_chunks.append({"source": title, "url": url, "content": piece})

        if not candidate_chunks or self.memory is None:
            return candidate_chunks[:2]

        q_vec = self.memory.embed_texts([query], is_query=True)[0]
        c_vecs = self.memory.embed_texts([c["content"] for c in candidate_chunks], is_query=False)
        scores = c_vecs @ q_vec
        top_idx = np.argsort(scores)[::-1][:2]

        best_chunks = []
        for i in top_idx:
            item = candidate_chunks[i]
            item["score"] = float(scores[i])
            best_chunks.append(item)
            self.memory.add_document(f"[{item['source']}] {item['content']}", source="web_cache")

        return best_chunks
