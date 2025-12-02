"""Train Temporal Convolutional Network (TCN) models on processed datasets.

This module exposes a programmatic API for training and evaluating a
TCN on preprocessed, padded time-series data saved as a PyTorch
checkpoint (the format produced by the repository's dataset helper).

Primary exports:
- ``train_tcn``: train a model (returns model, history, best_val)
- ``load_processed_dataset``: load the saved dataset checkpoint
- ``SequenceDataset``: thin Dataset wrapper for padded sequences

Behavior notes:
- Prefers saving ``state_dict()`` by default; full-model serialization
    is available via ``save_full_model=True``.
- Uses packed sequences to compute loss only on real (unpadded)
    timesteps and supports early stopping and LR scheduling.

"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

__contributors__ = ["Griffin C. Sipes", "Melany D. Opolz"]
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (c) 2025 The Board of Trustees of the University of Illinois"

# Module defaults (edit to suit your environment if you prefer)
DEFAULT_DATA_DIR = os.environ.get("TCN_DEFAULT_DATA_DIR", "/path/to/processed/data")
DEFAULT_OUT_DIR = os.environ.get("TCN_DEFAULT_OUT_DIR", "/path/to/output_dir")


def _safe_torch_load(path: str, map_location: Any = "cpu") -> Any:
    """Load a torch-saved file with helpful error messages.

    This function prefers to return simple python structures (dicts)
    or state_dict mappings. If a full ``nn.Module`` object is stored,
    it will be returned as-is (caller should validate/trust source).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        # prefer weights_only=False so metadata (scalers) is available
        return torch.load(path, map_location=map_location)
    except Exception as exc:  # pragma: no cover - runtime error
        # Retry without map_location for older torch versions
        try:
            return torch.load(path)
        except Exception as exc2:
            raise RuntimeError(f"Failed to load checkpoint {path}: {exc}; {exc2}")


def save_checkpoint_bundle(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    save_full_model: bool = False,
) -> None:
    """Save a checkpoint bundle containing model state, optimizer state and metadata.

    The saved file will be a dict with keys `model_state` and optionally
    `optimizer_state`, `epoch` and `meta`. If the checkpoint target is a
    plain state-dict (no wrapper), the function still writes the bundle to
    `path` to make later restores unambiguous.
    """
    bundle: Dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        try:
            bundle["optimizer_state"] = optimizer.state_dict()
        except Exception:
            # optimizer may not be initialized or serializable; skip gracefully
            pass
    if epoch is not None:
        bundle["epoch"] = int(epoch)
    if meta is not None:
        bundle["meta"] = dict(meta)

    torch.save(bundle, path)
    if save_full_model:
        # try to save a full-model object as convenience (best-effort)
        try:
            torch.save(model, os.path.splitext(path)[0] + "_full.pt")
        except Exception:
            pass


def load_checkpoint_bundle(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint bundle saved by :func:`save_checkpoint_bundle`.

    Returns a dict with whichever of the following keys exist:
    - ``model_state``: state_dict suitable for ``model.load_state_dict``
    - ``optimizer_state``: optimizer state dict
    - ``epoch``: epoch integer
    - ``meta``: user metadata dict
    - ``full_model``: an ``nn.Module`` if a full model object was saved

    If the file contains a raw state-dict (no wrapping dict), it will be
    returned as ``{'model_state': <state_dict>}`` for consistent handling.
    """
    ck = _safe_torch_load(path, map_location=map_location)
    if isinstance(ck, dict):
        # already a bundle or a raw state-dict
        # normalize raw state-dict -> {'model_state': raw}
        if any(k in ck for k in ("model_state", "optimizer_state", "epoch", "meta")):
            return dict(ck)
        else:
            return {"model_state": ck}
    elif isinstance(ck, torch.nn.Module):
        return {"full_model": ck}
    else:
        # unknown contents; wrap and return
        return {"model_state": ck}


class SequenceDataset(Dataset):
    """Dataset wrapper around padded sequence tensors.

    Expects a checkpoint (dict) with keys: ``X_*_padded``, ``y_*_padded``,
    and ``*_lengths``. This lightweight wrapper mirrors the earlier
    project's contract so it can be used directly with training loops.
    """

    def __init__(
        self,
        X_padded: torch.Tensor,
        y_padded: torch.Tensor,
        lengths: torch.Tensor,
        ids: Optional[Sequence[Any]] = None,
    ) -> None:
        self.X = X_padded
        self.y = y_padded
        self.lengths = lengths
        self.ids = ids

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[Any]]:
        return self.X[idx], self.y[idx], int(self.lengths[idx]), (self.ids[idx] if self.ids is not None else None)


def load_processed_dataset(data_dir: str) -> Dict[str, Any]:
    """Load a processed dataset checkpoint saved by the dataset creation helper.

    The function looks for ``dataset.pt`` inside either ``data_dir/Processed_Saved_Data``
    or directly in ``data_dir``. The returned object is the raw dict saved with
    :func:`torch.save` during preprocessing.
    """
    proc_dir = os.path.join(data_dir, "Processed_Saved_Data")
    pt_file = os.path.join(proc_dir, "dataset.pt")
    alt = os.path.join(data_dir, "dataset.pt")
    if os.path.exists(pt_file):
        return _safe_torch_load(pt_file)
    if os.path.exists(alt):
        return _safe_torch_load(alt)
    raise FileNotFoundError(f"Processed dataset not found at {pt_file} or {alt}")


class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-5) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss: Optional[float] = None
        self.counter = 0

    def __call__(self, loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = float(loss)
            return False
        if loss < self.best_loss - abs(self.best_loss) * self.min_delta:
            self.best_loss = float(loss)
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_tcn(
        data_dir: str = DEFAULT_DATA_DIR,
        out_dir: str = DEFAULT_OUT_DIR,
        device: str = "cpu",
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        num_filters: Optional[int] = None,
        num_layers: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        save_full_model: bool = False,
        quick_mode: bool = False,
        resume_from: Optional[str] = None,
        load_optimizer: bool = False,
        freeze_base: bool = False,
        freeze_prefixes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Train a TCN on a processed dataset and save best model/state dict.

    This function mirrors the original training API but adds optional
    support for resuming from a checkpoint and freezing early layers
    for fine-tuning.

    Parameters (new):
    - resume_from: optional path to a checkpoint (state_dict or dict)
        or a saved ``nn.Module`` to load into the model before training.
    - load_optimizer: if True and the checkpoint contains an
        ``optimizer_state`` mapping, the optimizer state will be restored.
    - freeze_base: if True, parameters whose names start with the
        prefixes in ``freeze_prefixes`` will be frozen (``requires_grad=False``).
    - freeze_prefixes: sequence of name prefixes to freeze when
        ``freeze_base`` is True. Defaults to common layer names.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if (device == "cuda" or (device == "auto" and torch.cuda.is_available())) else "cpu"

    # quick mode reduces epochs and batch size for smoke-testing
    if quick_mode:
        epochs = min(epochs, 3)
        batch_size = min(batch_size, 8)

    data = load_processed_dataset(data_dir)

    # prefer *_padded keys
    X_train = data.get("X_train_padded")
    y_train = data.get("y_train_padded")
    X_val = data.get("X_val_padded")
    y_val = data.get("y_val_padded")
    train_lengths = data.get("train_lengths")
    val_lengths = data.get("val_lengths")

    if X_train is None or y_train is None:
        raise KeyError("Processed dataset does not contain 'X_train_padded'/'y_train_padded'")

    # Construct datasets and dataloaders
    train_ds = SequenceDataset(X_train, y_train, train_lengths, ids=data.get("X_train_ids"))
    val_ds = SequenceDataset(X_val, y_val, val_lengths, ids=data.get("X_val_ids")) if X_val is not None else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    # infer sizes from first batch
    example = next(iter(train_loader))
    Xb, yb, lengths, _ = example
    input_size = Xb.shape[-1]
    output_size = yb.shape[-1]

    # default hyperparams mapping similar to legacy code
    if num_layers is None:
        num_layers = 3
    if num_filters is None:
        num_filters = 64

    # Import the TCN model locally to avoid circular imports
    try:
        from tcn_class import TCN_Model
    except Exception as exc:  # pragma: no cover - runtime import
        raise RuntimeError(f"Could not import TCN model class: {exc}")

    model = TCN_Model(
        input_size=int(input_size),
        num_filters=int(num_filters),
        num_layers=int(num_layers),
        output_size=int(output_size),
        kernel_size=int(kernel_size),
        dropout=float(dropout),
        dilation_base=int(dilation_base),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(l2_reg))
    criterion = nn.SmoothL1Loss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-8)
    early = EarlyStopping(patience=50) if not quick_mode else EarlyStopping(patience=10)

    # Optional resume/restore logic for fine-tuning. Use the centralized
    # bundle loader which normalizes raw state-dicts, bundles and full-model
    # objects into a predictable mapping.
    if resume_from is not None:
        bundle = load_checkpoint_bundle(resume_from, map_location=device)
        # prefer a full model object if present
        if "full_model" in bundle:
            model = bundle["full_model"].to(device)
        else:
            # accept multiple names for a state_dict for robustness
            ms = None
            for key in ("model_state", "state_dict", "model_state_dict"):
                if key in bundle:
                    ms = bundle[key]
                    break
            if ms is not None:
                try:
                    model.load_state_dict(ms)
                except Exception as exc:
                    raise RuntimeError(f"Loaded checkpoint state_dict could not be applied: {exc}")

            if load_optimizer and "optimizer_state" in bundle and opt is not None:
                try:
                    opt.load_state_dict(bundle["optimizer_state"])
                except Exception:
                    print("Warning: failed to restore optimizer state from checkpoint")

    # Optionally freeze base layers to fine-tune head only
    if freeze_base:
        prefixes = list(freeze_prefixes) if freeze_prefixes is not None else ["convs", "pad_layers"]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = False

    best_val = float("inf")
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for ep in range(int(epochs)):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for Xb, yb, lengths, _ in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device) if isinstance(lengths, torch.Tensor) else torch.tensor(lengths, device=device)

            # sanitize
            if torch.isnan(Xb).any() or torch.isinf(Xb).any():
                Xb = torch.nan_to_num(Xb, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(yb).any() or torch.isinf(yb).any():
                yb = torch.nan_to_num(yb, nan=0.0, posinf=1e6, neginf=-1e6)

            pred = model(Xb)
            if pred.size(1) != yb.size(1):
                L = min(pred.size(1), yb.size(1))
                pred = pred[:, :L, :]
                yb = yb[:, :L, :]
                lengths = torch.clamp(lengths, max=L)

            if (lengths == 0).any():
                nz = (lengths > 0).nonzero(as_tuple=True)[0]
                if nz.numel() == 0:
                    continue
                pred = pred[nz]
                yb = yb[nz]
                lengths = lengths[nz]

            pred_packed = pack_padded_sequence(pred, lengths.cpu(), batch_first=True, enforce_sorted=False)
            y_packed = pack_padded_sequence(yb, lengths.cpu(), batch_first=True, enforce_sorted=False)

            loss = criterion(pred_packed.data, y_packed.data)
            if l1_reg and l1_reg > 0.0:
                l1 = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        l1 = l1 + p.abs().sum()
                loss = loss + float(l1_reg) * l1

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            train_loss += float(loss.item())
            n_batches += 1

        train_loss = train_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # validation
        val_loss = 0.0
        n_val = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for Xb, yb, lengths, _ in val_loader:
                    Xb = Xb.to(device)
                    yb = yb.to(device)
                    lengths = lengths.to(device) if isinstance(lengths, torch.Tensor) else torch.tensor(lengths, device=device)

                    pred = model(Xb)
                    if pred.size(1) != yb.size(1):
                        L = min(pred.size(1), yb.size(1))
                        pred = pred[:, :L, :]
                        yb = yb[:, :L, :]
                        lengths = torch.clamp(lengths, max=L)

                    if (lengths == 0).any():
                        nz = (lengths > 0).nonzero(as_tuple=True)[0]
                        if nz.numel() == 0:
                            continue
                        pred = pred[nz]
                        yb = yb[nz]
                        lengths = lengths[nz]

                    pred_packed = pack_padded_sequence(pred, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    y_packed = pack_padded_sequence(yb, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    loss = criterion(pred_packed.data, y_packed.data)
                    val_loss += float(loss.item())
                    n_val += 1

            val_loss = val_loss / max(1, n_val)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
        else:
            val_loss = float("nan")

        # logging
        if not torch.isnan(torch.tensor(val_loss)):
            print(f"Epoch {ep+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, LR={opt.param_groups[0]['lr']:.6g}")
        else:
            print(f"Epoch {ep+1}/{epochs}: train_loss={train_loss:.6f}, LR={opt.param_groups[0]['lr']:.6g}")

        # checkpoint best
        metric = val_loss if not torch.isnan(torch.tensor(val_loss)) else train_loss
        if metric < best_val:
            best_val = float(metric)
            # Save a checkpoint bundle (model_state + optimizer_state + epoch + meta)
            bundle_path = os.path.join(out_dir, "best_checkpoint.pt")
            meta = {"feature_cols": data.get("feature_cols"), "target_cols": data.get("target_cols")}
            try:
                save_checkpoint_bundle(bundle_path, model, optimizer=opt, epoch=int(ep), meta=meta, save_full_model=save_full_model)
                print(f"Saved checkpoint bundle to: {bundle_path}")
            except Exception:
                # Fallback: save state_dict only
                state_path = os.path.join(out_dir, "tcn_model_state_dict.pt")
                torch.save(model.state_dict(), state_path)
                print(f"Warning: failed to save bundle; saved state_dict to: {state_path}")

            # Also save state_dict for backward compatibility
            try:
                state_path = os.path.join(out_dir, "tcn_model_state_dict.pt")
                torch.save(model.state_dict(), state_path)
            except Exception:
                pass

        # early stopping
        try:
            if early(val_loss if not torch.isnan(torch.tensor(val_loss)) else train_loss):
                print(f"Early stopping triggered at epoch {ep+1}")
                break
        except Exception:
            pass

    return {"model": model, "history": history, "best_val": best_val}


__all__ = ["train_tcn", "load_processed_dataset", "SequenceDataset", "DEFAULT_DATA_DIR", "DEFAULT_OUT_DIR"]


def main() -> None:
    """Run a full training job using module-level defaults.

    This function is intentionally non-interactive and does not parse
    CLI arguments; edit the module-level constants or modify the call
    below if you need different defaults.
    """
    data_dir = DEFAULT_DATA_DIR
    out_dir = DEFAULT_OUT_DIR

    # runtime defaults
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 200
    batch_size = 16
    lr = 1e-3

    print(f"Training TCN using data_dir={data_dir}, out_dir={out_dir}, device={device}")

    result = train_tcn(
        data_dir=data_dir,
        out_dir=out_dir,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        quick_mode=False,
        save_full_model=False,
    )

    best_val = result.get("best_val")
    print(f"Training complete. Best validation metric: {best_val}")
    print(f"State dict saved to: {os.path.join(out_dir, 'tcn_model_state_dict.pt')}")


if __name__ == "__main__":
    main()
