# Statistical approaches for estimating forelimb ground reaction forces (GRF)

Short, reproducible tools for estimating forelimb ground reaction forces (GRF) in foals using two statistical approaches: a simple linear regression model and a temporal convolutional network (TCN). This repository contains code, model artifacts, and example data to run both approaches for estimating peak and continuous GRF profiles from kinematic and mass data.

**Contributors:** Melany D. Opolz, Griffin C. Sipes, Sara G. Moshage, Annette M. McCoy, Mariana E. Kersh

## Table of Contents

- [High-level summary](#high-level-summary)
- [Repository layout](#repository-layout)
- [Requirements and recommended environment](#requirements-and-recommended-environment)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)
- [References and Citations](#references-and-citations)
- [License](#license)
- [Contact](#contact)

## High-level summary

- **Problem:** Measuring GRF in large-animal biomechanics is challenging; estimating GRF from kinematics enables wider studies of joint kinetics.
- **Approaches:** 1) Classical linear regression to estimate GRF peaks and profiles; 2) Machine-learning approach using a Temporal Convolutional Network (TCN) to estimate continuous GRF profiles.
- **Data:** Longitudinal motion capture, GRF recordings, and subject mass during walking and trotting gaits; Froude number used to account for subject size/dynamics.

## Repository layout

- `Linear Regression Model/` : Linear regression implementation and utilities (`grf_estimation_linear_model.py`).
- `Temporal Convolutional Network Model/` : TCN training and inference code (`tcn_class.py`, `tcn_runner.py`, `train_tcn_model.py`, `create_inference_dataset.py`).
- `Data/GRF Templates/` : Templates and example inputs for GRF processing.
- `Data/TCN Model Files/` : Example TCN model files and example input (`example_input-*.npz`, model checkpoints and metadata).
- `requirements.txt` : Python dependencies (use a virtual environment).

## Requirements and recommended environment

- Python 3.8 or newer. Use a virtual environment for isolation.

- Create and activate a virtual environment (pick the variant that matches your shell/OS):

```bash
# create venv (works on Windows/macOS/Linux; use `python` or `python3` as appropriate)
python -m venv .venv
```

Activation examples:

- Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

- Windows (Command Prompt)

```cmd
.venv\Scripts\activate.bat
```

- macOS / Linux (bash, zsh)

```bash
source .venv/bin/activate
```

- To exit the virtual environment:

```bash
deactivate
```

- Install required packages:

```bash
pip install -r requirements.txt
```

## File Structure

- `Linear Regression Model/grf_estimation_linear_model.py` : Main script for linear regression GRF estimation.
- `Temporal Convolutional Network Model/tcn_class.py` : TCN model definition.
- `Temporal Convolutional Network Model/tcn_runner.py` : TCN inference runner.
- `Temporal Convolutional Network Model/train_tcn_model.py` : Script to train the TCN model.
- `Temporal Convolutional Network Model/create_inference_dataset.py` : Prepares data for TCN inference.
- `Data/GRF Templates/` : Contains GRF templates.
- `Data/TCN Model Files/` : Contains TCN model files.

## Acknowledgements

The authors are grateful to Dr. Kellie Halloran for their assistance in data collection and code development. Funding support for this work was provided by Morris Animal Foundation D21EQ-004 and Grayson Jockey Club Research Foundation, Inc.

## References and Citations

If you use this code for research that results in publications, please cite our original article listed above.

You can use the following BibTeX entry

```bibtex
@article{Opolz:2025},
  title     = "Statistical approaches for estimating forelimb ground reaction forces in foals during walking and trotting",
  author    = "Melany D. Opolz, Griffin C. Sipes, Sara G. Moshage, Annette M. McCoy, and Mariana E. Kersh",
  year      = {2025},
  journal   = {Journal of Biomechanics}
}
```

## License

Copyright 2025 The Board of Trustees of the University of Illinois. All Rights Reserved.

Licensed under the terms of the Non-Exclusive Research Use license (the "License").

The License is included in the distribution as `License.txt` file.

You may not use these files except in compliance with the License.

Software distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  

See the License for the specific language governing permissions and limitations under the License.

## Contact

For questions about the code or issues, please open an issue on GitHub or contact the maintainers below.

Corresponding Author: Mariana E. Kersh (mkersh@illinois.edu)

Code Maintainer: Melany D. Opolz (mopolz@illinois.edu)