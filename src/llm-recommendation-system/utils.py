"""
Utility helpers for LLM recommendation system.
"""
from pathlib import Path
import pandas as pd

def load_data(filename: str = "/LLM-reco-sys/csv") -> pd.DataFrame:
    """Load CSV from data/ folder."""
    path = Path("data") / filename
    return pd.read_csv(path)
