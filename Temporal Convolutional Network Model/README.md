# TCN Inference & Training

## Description

This submodule contains utilities for inference, training and dataset
preparation for a Temporal Convolutional Network (TCN) used to predict
forelimb ground reaction forces (GRF).

## Table of Contents

- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [API Reference](#api-reference)
- [License](#license)

## Installation

Recommended Python: 3.8+ (3.9+ preferred). Install dependencies in a
virtual environment.

Create and activate a virtual environment (cross-platform). Use `python` or `python3` depending on your system.

POSIX (Linux / macOS):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows CMD:
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## File Structure

Top-level files in this submodule:

```
create_inference_dataset.py
tcn_class.py
tcn_runner.py
train_tcn_model.py
```

Short descriptions

- `tcn_class.py`: TCN implementation with `from_checkpoint` convenience loader.
- `train_tcn_model.py`: Programmatic training API (`train_tcn`), dataset loader and helpers.
- `tcn_runner.py`: Inference helpers (batched prediction, CSV/NPZ saving).
- `create_inference_dataset.py`: CSV -> padded `inference_dataset.pt` converter.

## Usage

Minimal examples showing typical workflows. These examples assume you
have created a virtual environment and installed the requirements.

Programmatic example to create an inference dataset from a feature CSV:

```py
# Preferred: call the module API if available
from pathlib import Path
import create_inference_dataset

create_inference_dataset.create_inference_dataset(
  csv_path=Path("path/to/features.csv"),
  out_path=Path("path/to/Processed_Saved_Data/inference_dataset.pt"),
  pad_length=1000,  # optional, adjust to your preprocessing needs
)

# Fallback: invoke the script entrypoint (if it exposes a main() that accepts arg list)
import create_inference_dataset as cid
cid.main([
  "--in-csv", "path/to/features.csv",
  "--out-dir", "path/to/Processed_Saved_Data",
  "--pad-length", "1000",
])
```

Programmatic example to load a checkpoint and run inference:

```py
from pathlib import Path
import torch
import numpy as np
from tcn_class import TCN_Model
import tcn_runner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load checkpoint into model (state-dict or full model supported)
model = TCN_Model.from_checkpoint('path/to/best_checkpoint.pt', device)

# load prepared dataset
data = torch.load('path/to/Processed_Saved_Data/inference_dataset.pt', map_location='cpu')
X = data['X_test_padded']  # (N, T, features)

# run batched predictions
preds = tcn_runner.predict_on_array(model, X, device=device, batch_size=32)

# save compressed outputs
tcn_runner.save_predictions_npz('out/preds.npz', {'preds': preds})
```

Train a model programmatically:

```py
from train_tcn_model import train_tcn

train_tcn(
    data_dir=r"G:\path\to\Processed_Saved_Data",
    out_dir=r"G:\path\to\Model Outputs\TCN",
    epochs=50,
)
```

## API Reference

Key helpers (importable):

- `tcn_class.TCN_Model.from_checkpoint(checkpoint_path, device, **model_kwargs)` — load model from file (accepts state-dict bundles or full `nn.Module`).
- `tcn_runner.predict_on_array(model, X, device=None, batch_size=32)` — run inference on `(N,T,F)` arrays, returns `(N,T,outputs)` numpy array.
- `tcn_runner.save_predictions_npz(path, save_dict)` — save compressed NPZ outputs.
- `train_tcn_model.train_tcn(...)` — programmatic training loop with checkpointing.

See docstrings in the modules for detailed parameter descriptions and
return values.

## License

This submodule uses the GPL-3.0-or-later license as declared in module
level `__license__` attributes. See the `LICENSE` file for the SPDX
identifier and copyright notice.