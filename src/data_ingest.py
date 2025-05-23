import pandas as pd
from pathlib import Path
from tqdm import tqdm

TEXT_COLS = [
    "title_en",
    "abstract_text",
    "claims",
    "description_text",
    "analysis_explanation"
]

def load_csv(csv_path: Path, save_parquet: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} patents")
    if save_parquet:
        p_path = csv_path.with_suffix(".parquet")
        df.to_parquet(p_path, index=False)
        print(f"✅ Parquet saved → {p_path}")
    return df

def concat_text(row, cols=TEXT_COLS, sep="\n\n"):
    return sep.join(str(row[c] or "") for c in cols if pd.notna(row[c]))

# --- simple token/word count utility -------------------------
from .token_utils import count_tokens, count_words

def text_stats(df: pd.DataFrame, cols=TEXT_COLS):
    """Print avg / max token and word counts across selected columns."""
    token_counts, word_counts = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        txt = concat_text(row, cols)
        token_counts.append(count_tokens(txt))
        word_counts.append(count_words(txt))
    print(f"Avg tokens/patent: {sum(token_counts)//len(token_counts)}")
    print(f"Max tokens: {max(token_counts)}, 90th-pct: {int(pd.Series(token_counts).quantile(0.9))}")
