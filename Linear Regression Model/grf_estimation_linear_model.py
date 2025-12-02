"""Equine GRF estimation (linear-template model).

This module provides utilities to estimate forelimb ground reaction forces
(GRFs) for foals using a simple linear-template approach. Given limb length,
velocity, and body mass, the module computes the Froude number, selects a
walk/trot template, scales the template according to regression-derived peak
predictions, and writes the resulting anterior-posterior (Fy) and vertical
(Fz) forces to CSV and PNG files.

Primary public functions
 - :func:`generate_grf` — produce scaled Fy and Fz arrays from templates.
 - :func:`save_outputs` — save CSV and PNG outputs for given Fy/Fz arrays.

Notes
 - Expected template filenames: ``TemplateGRFwalk.xlsx`` and
   ``TemplateGRFtrot.xlsx`` located in ``TEMPLATE_DIR`` (module-level default).
"""


import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

__contributors__ = ["Melany D. Opolz", "Griffin C. Sipes"]
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (c) 2025 The Board of Trustees of the University of Illinois"


WALK_TEMPLATE = "TemplateGRFwalk.xlsx"
TROT_TEMPLATE = "TemplateGRFtrot.xlsx"


def compute_froude(velocity: float, limb_length: float) -> float:
    """Compute the Froude number.

    Parameters
    ----------
    velocity : float
        Velocity in metres per second.
    limb_length : float
        Limb length in metres.

    Returns
    -------
    float
        The Froude number: v**2 / (g * limb_length).
    """
    return (velocity ** 2) / (limb_length * 9.81)


def select_template_and_scales(fr: float) -> Tuple[str, float, float, float, int]:
    """Select walk/trot template and compute scaling factors.

    Parameters
    ----------
    fr : float
        Froude number.

    Returns
    -------
    tuple
        A tuple of (template_filename, Fz_max, Fy_max, Fy_min, split_index).

    Notes
    -----
    The ``split_index`` separates the A-P minima and maxima rows in the
    template files. The values mirror the original script's indexing
    (walk: 364, trot: 139).
    """
    if fr < 0.5:
        # Walk
        Fz_max = 4.91 * fr + 5.49
        Fy_max = 2.11 * fr + 0.47
        Fy_min = abs(0.30 * fr - 0.90)
        return WALK_TEMPLATE, Fz_max, Fy_max, Fy_min, 364

    # Trot
    Fz_max = 1.03 * fr + 11.29
    Fy_max = 0.29 * fr + 0.59
    Fy_min = abs(-0.30 * fr - 0.95)
    return TROT_TEMPLATE, Fz_max, Fy_max, Fy_min, 139


def load_template(template_path: Path, split_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load a template Excel file and return template arrays.

    Parameters
    ----------
    template_path : Path
        Path to the Excel template file.
    split_index : int
        Row index at which to split the A-P minima/maxima sections.

    Returns
    -------
    tuple
        ``(old_ymin, old_ymax, old_z)`` where ``old_ymin`` and ``old_ymax`` are
        1-D numpy arrays of the anterior-posterior template values and
        ``old_z`` is the vertical template array.
    """
    df = pd.read_excel(template_path)
    old_ymin = df.iloc[:split_index, 0].values
    old_ymax = df.iloc[split_index:, 0].values
    old_z = df.iloc[:, 1].values
    return (old_ymin, old_ymax, old_z)


def generate_grf(limb_length: float, velocity: float, mass: float,
                 template_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Generate scaled anterior-posterior (Fy) and vertical (Fz) forces.

    Parameters
    ----------
    limb_length : float
        Limb length in metres.
    velocity : float
        Velocity in metres per second.
    mass : float
        Body mass in kilograms.
    template_dir : Path
        Directory containing the template Excel files.

    Returns
    -------
    tuple
        ``(new_Fy, new_Fz)`` — two 1-D numpy arrays containing the scaled
        anterior-posterior and vertical forces, respectively.

    Raises
    ------
    FileNotFoundError
        If the expected template file is not found in ``template_dir``.
    """
    fr = compute_froude(velocity, limb_length)
    template_file, Fz_max, Fy_max, Fy_min, split_index = select_template_and_scales(fr)

    tpl_path = template_dir / template_file
    if not tpl_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {tpl_path}. "
            "Place template files in the provided template directory or use --template-dir"
        )

    old_ymin, old_ymax, old_z = load_template(tpl_path, split_index)

    new_Fz = (old_z * Fz_max) * mass
    new_Fy = np.concatenate((old_ymin * Fy_min, old_ymax * Fy_max)) * mass

    return new_Fy, new_Fz


def save_outputs(new_Fy: np.ndarray, new_Fz: np.ndarray, output_dir: Path, file_name: str,
                 show_plot: bool = True) -> Tuple[Path, Path]:
    """Save Fy/Fz arrays to CSV and PNG files.

    Parameters
    ----------
    new_Fy : np.ndarray
        Array of anterior-posterior forces.
    new_Fz : np.ndarray
        Array of vertical forces.
    output_dir : Path
        Directory to write output files; created if it does not exist.
    file_name : str
        Base filename (without extension) for output files.
    show_plot : bool, optional
        If True, display the matplotlib plot (default: True).

    Returns
    -------
    tuple
        Paths to the saved CSV and PNG files: ``(output_csv, output_png)``.
    """
    force_data = np.column_stack((new_Fy, new_Fz))
    headers = "Fy, Fz"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{file_name}.csv"
    np.savetxt(output_csv, force_data, header=headers, delimiter=",",
               comments='', fmt="%.6f")

    percent_stance = np.linspace(0, 100, len(new_Fy))

    plt.figure(figsize=(7, 5))
    plt.plot(percent_stance, new_Fy, label="A-P (Fy)")
    plt.plot(percent_stance, new_Fz, label="Vertical (Fz)")
    plt.xlabel("% Stance")
    plt.ylabel("Force (N)")
    plt.title("Generated GRF Data")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 100)

    output_png = output_dir / f"{file_name}.png"
    plt.savefig(output_png, bbox_inches='tight', dpi=300, facecolor='white')
    if show_plot:
        plt.show()
    plt.close()

    return output_csv, output_png


# ----------------------------- User-Defined Inputs ----------------------------
# NOTE: These inputs will need to be changed to suit your needs

# Filepaths
# Users should update `TEMPLATE_DIR` and `OUTPUT_DIR` to point to the
# folder containing `TemplateGRFwalk.xlsx` and `TemplateGRFtrot.xlsx` and
# the desired location for output CSV/PNG files.
TEMPLATE_DIR = Path("/Pathto/templates")
OUTPUT_DIR = Path("/Pathto/output")

# Defaults for the GRF generation (change as needed)
LIMB_LENGTH = 0.78  # meters
VELOCITY = 3.0      # m/s
MASS = 300.0        # kg
FILE_NAME = "GeneratedGRF"
# Show plot on completion (set False for headless runs)
SHOW_PLOT = True


def main() -> None:
    print("Using template directory:", TEMPLATE_DIR)
    print("Using output directory:", OUTPUT_DIR)
    print("Expected template filenames:", WALK_TEMPLATE, ",", TROT_TEMPLATE)
    print("To change paths, edit TEMPLATE_DIR and OUTPUT_DIR at the top of this file.")

    new_Fy, new_Fz = generate_grf(
        limb_length=LIMB_LENGTH,
        velocity=VELOCITY,
        mass=MASS,
        template_dir=Path(TEMPLATE_DIR),
    )

    csv_path, png_path = save_outputs(new_Fy, new_Fz, Path(OUTPUT_DIR), FILE_NAME,
                                      show_plot=SHOW_PLOT)

    print("GRF estimation complete.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()



