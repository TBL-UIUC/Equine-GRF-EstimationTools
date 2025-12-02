"""Utilities to load a TCN model, run predictions and save outputs.

This module provides small, testable helpers that orchestrate model
loading and inference. The neural network implementation remains in
``tcn_class.py``; this runner handles dataset loading, batching and
basic saving of numpy outputs.

Helpers in this module are intentionally small so they can be used in
other scripts or unit tests without bringing the full training
pipeline into scope.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tcn_class import TCN_Model

__contributors__ = ["Griffin C. Sipes", "Melany D. Opolz"]
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (c) 2025 The Board of Trustees of the University of Illinois"


def load_model_from_checkpoint(
    checkpoint_path: str, device: Union[str, torch.device] = "cpu", **model_kwargs: Any
) -> torch.nn.Module:
    """Load a TCN model from a checkpoint.

    The function delegates to ``TCN_Model.from_checkpoint``. If the
    checkpoint contains only a state dict, ``model_kwargs`` must be
    provided to construct the template model instance.
    """
    return TCN_Model.from_checkpoint(checkpoint_path, device=device, **model_kwargs)


def predict_on_array(
    model: torch.nn.Module,
    X: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Run model inference on a batched numpy array.

    Parameters
    ----------
    model : torch.nn.Module
        The model in eval mode.
    X : np.ndarray
        Input array with shape ``(N, T, features)``.
    device : str or torch.device, optional
        Device to run inference on. If ``None``, use model device.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    np.ndarray
        Predictions with shape ``(N, T, outputs)``.
    """
    # determine device from model if not provided
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    if device is None:
        device = model_device
    device = torch.device(device)

    model = model.to(device).eval()

    # Accept tensors or numpy arrays; ensure float32
    X_arr = np.asarray(X, dtype=np.float32)

    N = int(X_arr.shape[0])
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_arr[i : i + batch_size]
            batch_t = torch.from_numpy(batch).to(device)
            out = model(batch_t)
            preds.append(out.cpu().numpy())

    if len(preds) == 0:
        return np.empty((0, 0, 0), dtype=np.float32)

    return np.vstack(preds)


def save_predictions_npz(out_path: str, save_dict: Dict[str, Any]) -> None:
    """Ensure output directory exists and save a compressed NPZ file.

    The ``save_dict`` is forwarded to ``numpy.savez_compressed``.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)


def _write_model_csv(
    csv_path: str,
    y_array: np.ndarray,
    pred_array: np.ndarray,
    ids: Optional[Sequence[Union[str, int]]] = None,
    lengths: Optional[Sequence[int]] = None,
    target_colnames: Optional[Sequence[str]] = None,
) -> None:
    """Write per-timepoint predictions and ground-truth to ``csv_path``.

    Both ``y_array`` and ``pred_array`` are expected shaped ``(N, T, C)``.
    ``lengths`` contains the true per-trial lengths (T_i) to avoid
    iterating over padded timesteps. ``ids`` may be a sequence of
    identifiers for each trial.
    """
    N, T, C = int(y_array.shape[0]), int(y_array.shape[1]), int(y_array.shape[2])
    if lengths is None:
        lengths = [T] * N

    # normalize ids into strings
    if ids is None:
        ids_list = [""] * N
    else:
        ids_list = [str(x) for x in list(ids)]

    # prepare headers
    if target_colnames is not None and len(target_colnames) >= C:
        y_cols = [f"y_{name}" for name in target_colnames[:C]]
        p_cols = [f"pred_{name}" for name in target_colnames[:C]]
    else:
        y_cols = [f"y_{i}" for i in range(C)]
        p_cols = [f"pred_{i}" for i in range(C)]

    fieldnames = ["trial_index", "trial_id", "time_index"] + y_cols + p_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            L = int(lengths[i]) if i < len(lengths) else T
            for t in range(L):
                row: Dict[str, Any] = {"trial_index": i, "trial_id": ids_list[i], "time_index": t}
                for c in range(C):
                    row[y_cols[c]] = float(y_array[i, t, c])
                    row[p_cols[c]] = float(pred_array[i, t, c])
                writer.writerow(row)


def save_predictions_csv(
    out_dir: str,
    y_array: Optional[np.ndarray],
    pred_array: np.ndarray,
    ids: Optional[Sequence[Union[str, int]]] = None,
    lengths: Optional[Sequence[int]] = None,
    target_colnames: Optional[Sequence[str]] = None,
    model_name: str = "tcn",
) -> None:
    """Save predictions and ground-truth as CSVs in `out_dir`.

    Creates a file named `preds_{model_name}.csv` containing per-timepoint
    rows with trial index, optional id, time index, ground-truth channels
    and prediction channels.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"preds_{model_name}.csv")
    _write_model_csv(csv_path, y_array if y_array is not None else np.zeros_like(pred_array), pred_array, ids=ids, lengths=lengths, target_colnames=target_colnames)
    print(f"Wrote {model_name.upper()} predictions CSV to {csv_path}")


def main() -> None:
    # === Default parameters (replace these or set environment variables) ===
    # For convenience you can set environment variables `TCN_DATA_DIR`,
    # `TCN_CHECKPOINT` and `TCN_OUT_DIR` to override the defaults.
    data_dir = os.environ.get("TCN_DATA_DIR", os.path.join(os.getcwd(), "Processed_Saved_Data"))
    checkpoint = os.environ.get("TCN_CHECKPOINT", os.path.join(os.getcwd(), "tcn_model.pt"))
    out_dir = os.environ.get("TCN_OUT_DIR", os.path.join(os.getcwd(), "model_outputs"))

    batch_size = int(os.environ.get("TCN_BATCH_SIZE", "32"))
    device = os.environ.get("TCN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Lazy import to avoid forcing a dependency for callers that only use helpers
    try:
        from train_tcn_model import load_processed_dataset
    except Exception as exc:  # pragma: no cover - runtime import error
        raise RuntimeError(f"Could not import train_tcn_model.load_processed_dataset: {exc}")

    data = load_processed_dataset(data_dir)
    X_test = data['X_test_padded']
    y_test = data.get('y_test_padded')

    N, T, in_ch = int(X_test.shape[0]), int(X_test.shape[1]), int(X_test.shape[2])
    out_ch = int(y_test.shape[2]) if y_test is not None else None

    model_kwargs: Dict[str, int] = {"input_size": in_ch}
    if out_ch is not None:
        model_kwargs['output_size'] = out_ch

    model = load_model_from_checkpoint(checkpoint, device=device, **model_kwargs)

    preds = predict_on_array(model, X_test, device=device, batch_size=batch_size)

    save_dict: Dict[str, Any] = {
        "preds_tcn_scaled": preds,
        "y_test_scaled": np.array(y_test) if y_test is not None else None,
    }

    # determine ids and lengths & target column names if present
    ids_for_csv = data.get("X_test_ids") if "X_test_ids" in data else None
    Ls: Optional[np.ndarray] = None
    if "test_target_lengths" in data:
        Ls = np.array(data.get("test_target_lengths"))
    elif "test_lengths" in data:
        Ls = np.array(data.get("test_lengths"))
    target_cols = data.get("target_cols") or None

    # prefer physical units if present in caller's postprocessing; here we have scaled preds
    y_for_csv = np.array(y_test) if y_test is not None else None

    # write CSV for TCN predictions
    try:
        save_predictions_csv(
            out_dir,
            y_for_csv,
            preds,
            ids=ids_for_csv,
            lengths=Ls,
            target_colnames=target_cols,
            model_name="tcn",
        )
    except Exception as exc:
        print("Failed to write TCN CSV:", exc)


__all__ = [
    "load_model_from_checkpoint",
    "predict_on_array",
    "save_predictions_npz",
    "save_predictions_csv",
]


if __name__ == "__main__":
    main()
