"""Prepare an inference dataset compatible with the pretrained TCN model.

This script is a lightweight, documented copy of the original
`CreateTrainingDataset.py` focused only on formatting user-provided
feature CSVs into a padded, per-trial checkpoint that the TCN runner
(`tcn_runner.py`) can load for inference.

Edit the hard-coded paths below to point to your input CSV and desired
output folder, then run the script. If you have a saved `X` scaler
(joblib) from training, set `X_SCALER_PATH` so the same scaling is
applied before padding (recommended if the pretrained model expects
scaled inputs).

Output checkpoint keys (torch.save):
- `X_test_padded` : Tensor (N, T, features)
- `X_test_ids` : list[str]
- `test_lengths` : Tensor of per-trial lengths
- `feature_cols` : list of column names used

Expected input CSV format:
- Contains a column named `ID` that groups rows belonging to the same
    trial/sample.
- Optional `Time` column may be provided to sort rows within each trial;
    if absent, row order is preserved.
- Required input feature columns (order must match the model's expected input):
    - Mass
    - LegLength
    - FroudeNumber
    - VelocityX
    - VelocityY
    - VelocityZ
    - AccelerationX
    - AccelerationY
    - AccelerationZ

Extra columns in the CSV are ignored. The script will raise an error if
any of the required feature columns above are missing.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None
import torch
from torch.nn.utils.rnn import pad_sequence

__contributors__ = ["Griffin C. Sipes", "Melany D. Opolz"]
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (c) 2025 The Board of Trustees of the University of Illinois"

# --- User-configurable defaults ---
# NOTE: Replace these defaults with your own paths before running the script.
# You may also set the environment variables `TCN_INPUT_CSV`, `TCN_OUT_DIR`
# and `TCN_X_SCALER_PATH` to override the defaults at runtime.
INPUT_CSV = os.environ.get("TCN_INPUT_CSV", "/path/to/new_features.csv")
OUT_DIR = os.environ.get("TCN_OUT_DIR", "/path/to/output/Processed_Saved_Data")

# If you have a joblib scaler from training, set its path here or via
# the `TCN_X_SCALER_PATH` environment variable; otherwise leave None.
X_SCALER_PATH: Optional[str] = os.environ.get("TCN_X_SCALER_PATH", None)  # e.g. r'path/to/feature_scaler_train.joblib'

# Columns expected by the pretrained TCN model. Keep order consistent.
FEATURE_COLS: List[str] = [
    "Mass",
    "LegLength",
    "FroudeNumber",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "AccelerationX",
    "AccelerationY",
    "AccelerationZ",
]
# Optional time column name (used for sorting rows within each trial)
TIME_COL = "Time"


def _normalize_id(x) -> str:
    """Normalize an ID value to a stripped string.

    Returns an empty string for missing values.
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_features_table(path: str) -> pd.DataFrame:
    """Load a features table from CSV or Excel.

    Parameters
    ----------
    path : str
        Path to the input file. CSV is chosen when the filename ends with
        ``.csv`` (case-insensitive); otherwise ``read_excel`` is used.

    Returns
    -------
    pandas.DataFrame
        The loaded table.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce columns in ``cols`` to numeric, setting non-convertible
    entries to NaN.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def group_to_tensor_list(
    df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    time_col: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Group rows by ``id_col`` and convert each group to a tensor.

    The returned tensors have shape ``(T, features)``.
    """
    # normalize IDs
    df[id_col] = df[id_col].apply(_normalize_id)

    # Ensure required feature columns present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in input CSV: {missing}")

    groups: list[torch.Tensor] = []
    ids: list[str] = []
    for gid, gdf in df.groupby(id_col, sort=False):
        if time_col and time_col in gdf.columns:
            gdf = gdf.sort_values(time_col)
        arr = gdf[feature_cols].to_numpy(dtype=float)
        # convert to torch tensor (T, features)
        t = torch.from_numpy(arr.astype(np.float32))
        groups.append(t)
        ids.append(gid)

    return groups, ids


def apply_scaler_to_list(tensor_list: List[torch.Tensor], scaler) -> List[torch.Tensor]:
    """Apply a scikit-learn scaler to a list of sequence tensors.

    The scaler is applied to the flattened concatenation of all sequences
    then split back into the original per-sequence shapes.
    """
    # scaler.transform expects 2D array; we flatten, transform, and reshape
    all_flat = np.vstack([t.numpy() for t in tensor_list])
    transformed = scaler.transform(all_flat)

    # now split back according to original lengths
    out_list: list[torch.Tensor] = []
    idx = 0
    for t in tensor_list:
        L = t.shape[0]
        seg = transformed[idx : idx + L]
        out_list.append(torch.from_numpy(seg.astype(np.float32)))
        idx += L
    return out_list


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Loading input CSV:', INPUT_CSV)
    df = load_features_table(INPUT_CSV)

    if 'ID' not in df.columns:
        raise KeyError("Input CSV must contain an 'ID' column that groups rows per trial/sample")

    # coerce numeric feature columns
    df = coerce_numeric(df, FEATURE_COLS + ([TIME_COL] if TIME_COL in df.columns else []))

    # Build per-trial tensor list
    X_list, X_ids = group_to_tensor_list(
        df, "ID", FEATURE_COLS, TIME_COL if TIME_COL in df.columns else None
    )

    if len(X_list) == 0:
        raise RuntimeError('No trials found in input CSV')

    # Optionally load scaler
    if X_SCALER_PATH is not None:
        if joblib is None:
            raise RuntimeError('joblib is required to load scaler but is not installed')
        if not os.path.exists(X_SCALER_PATH):
            raise FileNotFoundError(X_SCALER_PATH)
        scaler = joblib.load(X_SCALER_PATH)
        X_list = apply_scaler_to_list(X_list, scaler)

    # Pad sequences to create (N, T, features)
    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([t.shape[0] for t in X_list], dtype=torch.long)

    save_path = os.path.join(OUT_DIR, 'inference_dataset.pt')
    save_dict = {
        "X_test_padded": X_padded,
        "X_test_ids": X_ids,
        "test_lengths": lengths,
        "feature_cols": FEATURE_COLS,
    }
    torch.save(save_dict, save_path)
    print('Saved inference dataset to:', save_path)

    # For compatibility with `load_processed_dataset` which looks for
    # `dataset.pt` inside the processed folder, also save a copy named
    # `dataset.pt` so downstream runners can find it without renaming.
    dataset_copy = os.path.join(OUT_DIR, "dataset.pt")
    try:
        torch.save(save_dict, dataset_copy)
        print('Also saved compatibility dataset to:', dataset_copy)
    except Exception:
        # non-fatal; inference_dataset.pt is the canonical file produced here
        pass


if __name__ == "__main__":
    main()
