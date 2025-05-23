import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Sequence
from .config import EMB_MODEL_NAME, EMB_DIR
from .filter_ops import apply_filter


class PassageRetriever:
    def __init__(self,
                 df: pd.DataFrame | None = None,
                 index_name: str = "faiss_chunks.idx"):
        # 1) load metadata table (DataFrame) if not provided
        if df is None:
            pq = EMB_DIR / "patents.parquet"
            pk = EMB_DIR / "patents.pkl"
            if pq.exists():
                try:
                    df = pd.read_parquet(pq)
                except ImportError:
                    # no parquet engine installed
                    df = pd.read_pickle(pk)
            elif pk.exists():
                df = pd.read_pickle(pk)
            else:
                raise FileNotFoundError(
                    f"Neither {pq} nor {pk} found – please run embed_build.py"
                )
        self.df = df

        # 2) load FAISS index & chunk meta
        idx_path = EMB_DIR / index_name
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {idx_path}")

        import faiss, pickle
        self.index = faiss.read_index(str(idx_path))
        with open(EMB_DIR / "meta.pkl", "rb") as f:
            self.meta = pickle.load(f)

        # 3) init encoder for on-the-fly queries

        self.model = SentenceTransformer(EMB_MODEL_NAME)


    # ------------- helpers ------------------------------------------------
    def _row_matches(self, row: pd.Series, chunk: str,
                     filters: Sequence[Dict[str, Any]]) -> bool:
        for f in filters:
            col, op, val = f["column"], f["op"], f["value"]
            target = chunk if col == "_chunk_text" else row.get(col, "")
            if not apply_filter(target, op, val):
                return False
        return True

    # ------------- public search -----------------------------------------
    def search(self, query: str,
               max_passages: int = 400,
               filters: Sequence[Dict[str, Any]] | None = None,
               column_order: List[str] | None = None,
               top_k_return: int = 60) -> List[Dict[str, Any]]:

        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I  = self.index.search(q_emb, max_passages)

        hits = []
        for idx, score in zip(I[0], D[0]):
            meta = self.meta[idx]
            row  = self.df.iloc[meta["row_idx"]]
            if filters and not self._row_matches(row, meta["chunk_text"], filters):
                continue
            hits.append({
                "publication_number": str(meta["publication_number"]),
                "title": str(row.get("title_en", "")),
                "text":  meta["chunk_text"],
                "row":   row,
                "vec_score": float(score)
            })

        # keep only 1st chunk per patent to diversify
        seen = set()
        uniq = []
        for h in hits:
            pid = h["publication_number"]
            if pid not in seen:
                seen.add(pid)
                uniq.append(h)
        hits = uniq

        # simple re-rank bonus for earlier column_priority matches
        if column_order:
            weight = {c: (len(column_order) - i) for i, c in enumerate(column_order)}

            def bonus(hit):
                b = 0.0
                row = hit["row"]
                for col in column_order:
                    cell = str(row.get(col, "")).lower()
                    if cell and any(t in cell for t in query.lower().split()):
                        b += weight[col]
                return b

            for h in hits:
                h["score"] = -h["vec_score"] + bonus(h)
            hits.sort(key=lambda x: x["score"], reverse=True)
        else:
            hits.sort(key=lambda x: x["vec_score"])

        return [{k: h[k] for k in ("publication_number", "title", "text")}
                for h in hits[:top_k_return]]
