import pandas as pd
from collections import Counter
from typing import Dict

def top_k_group(df: pd.DataFrame, column: str, k: int = 10) -> Dict[str,int]:
    """
    Return the k most frequent values in <column>, as {value: count}.
    Splits semi-colon or comma-separated entries.
    """
    series = (
        df[column]
        .dropna()
        .astype(str)
        .str.split(r"[;,|]")
        .explode()
        .str.strip()
        .value_counts()
        .head(k)
    )
    return series.to_dict()

def group_by_year(df: pd.DataFrame, date_col: str) -> Dict[int,int]:
    """
    Group df by the year of <date_col>, return {year:count}, sorted ascending.
    """
    years = pd.to_datetime(df[date_col], errors="coerce").dt.year.dropna().astype(int)
    counts = years.value_counts().sort_index()
    return counts.to_dict()
