import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Sequence, Union
import os
import sys

# Add the project root to the Python path if the module isn't found
try:
    from config import EMB_MODEL_NAME, EMB_DIR
except ModuleNotFoundError:
    # If running as module from src, use relative import
    try:
        from src.config import EMB_MODEL_NAME, EMB_DIR
    except ModuleNotFoundError:
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.config import EMB_MODEL_NAME, EMB_DIR

try:
    from filter_ops import apply_filter
except ModuleNotFoundError:
    try:
        from src.filter_ops import apply_filter
    except ModuleNotFoundError:
        # Define filter function inline to remove dependency
        def apply_filter(row_val, op, value):
            """Apply filter operation on row value."""
            row_text = str(row_val).lower()
            
            if op == "eq":
                return str(row_val) == str(value)
            elif op == "neq":
                return str(row_val) != str(value)
            elif op == "contains":
                return str(value).lower() in row_text
            elif op == "startswith":
                return row_text.startswith(str(value).lower())
            elif op == "in":
                if isinstance(value, (list, tuple)):
                    return any(str(v).lower() in row_text for v in value)
                return str(value).lower() in row_text
            else:
                return True  # Unknown operation, don't filter


class PassageRetriever:
    def __init__(self,
                 df: Union[pd.DataFrame, None] = None,
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
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔥 PassageRetriever using device: {device}")
        self.model = SentenceTransformer(EMB_MODEL_NAME, device=device)


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
               filters: Union[Sequence[Dict[str, Any]], None] = None,
               column_order: Union[List[str], None] = None,
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
