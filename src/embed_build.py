# Updated src/embed_build.py to gracefully handle missing parquet engine

from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss, pickle, numpy as np, pandas as pd
from tqdm import tqdm
import sys

from .config import EMB_MODEL_NAME, EMB_DIR
from .data_ingest import concat_text, TEXT_COLS
from .token_utils import count_tokens

def iter_chunks(text: str, max_tokens: int = 512, overlap: int = 64):
    words = text.split()
    step  = max_tokens - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + max_tokens])

def build_index(df: pd.DataFrame,
                cols       = None,
                index_name = "faiss_chunks.idx"):
    """
    Build FAISS index on text chunks (for fine-grained recall),
    and persist both the index and the original DataFrame.
    """
    model   = SentenceTransformer(EMB_MODEL_NAME)
    all_embs, meta = [], []

    print("🔨  Encoding chunks …")
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        use_cols  = cols or TEXT_COLS
        full_text = concat_text(row, cols=use_cols)
        if not full_text.strip():  # skip empty
            continue
        for chunk_id, chunk in enumerate(iter_chunks(full_text)):
            emb = model.encode(chunk, convert_to_numpy=True)
            all_embs.append(emb)
            meta.append({
                "row_idx": row_idx,
                "chunk_id": chunk_id,
                "publication_number": row["publication_number"],
                "chunk_text": chunk
            })

    if not all_embs:
        raise ValueError("No text found to index -- check column names!")

    embs_np = np.vstack(all_embs).astype("float32")
    dim     = embs_np.shape[1]
    index   = faiss.IndexFlatL2(dim)
    index.add(embs_np)

    # ensure directory exists
    EMB_DIR.mkdir(exist_ok=True)

    # 1) save FAISS index
    idx_path = EMB_DIR / index_name
    faiss.write_index(index, str(idx_path))

    # 2) save the metadata for each chunk
    with open(EMB_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # 3) persist the full patent DataFrame once, fallback to pickle if parquet unavailable
    try:
        df.to_parquet(EMB_DIR / "patents.parquet", index=False)
        print(f"✅ Full DataFrame saved → {EMB_DIR/'patents.parquet'}")
    except (ImportError, ValueError):
        print("⚠️  pyarrow/fastparquet not available, saving DataFrame as pickle instead")
    df.to_pickle(EMB_DIR / "patents.pkl")
    print(f"✅ Full DataFrame saved → {EMB_DIR/'patents.pkl'}")

    print(f"✅ FAISS index saved ({len(meta):,} chunks) → {idx_path}")
    print(f"✅ Metadata saved → {EMB_DIR/'meta.pkl'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.embed_build <path/to/final_dataset.csv>")
        sys.exit(1)
    csv_path = sys.argv[1]
    print(f"Loading CSV from {csv_path} …")
    df = pd.read_csv(csv_path)
    build_index(df)
