# Generalizable Contact Shape Reconstruction using EIT-based Tactile Sensing

Research code accompanying the paper **"Generalizable Contact Shape Reconstruction using Electrical Impedance Tomography (EIT)-based Tactile Sensing."**

This repository investigates contact sensing with a soft EIT tactile sensor. It contains two related pipelines:

1. **Contact-shape reconstruction** from EIT voltage measurements, including simulation, real-data processing, model-based reconstruction, and learned reconstruction models.
2. **Simulation-to-real double-touch localization**, where a conditional diffusion model translates simulated EIT signals toward the real-sensor domain before estimating two contact locations.

The repository contains research scripts rather than a packaged library. Several scripts expect datasets, trained weights, an EIT acquisition board, or a UR robot that are not included in the source repository.

## Repository structure

```text
.
|-- eit_shape_reconstruction/
|   |-- sim/       # Synthetic EIT data generation, training, evaluation, figures
|   |-- real/      # Real-data preparation, fine-tuning, testing, visualization
|   |-- robot/     # EIT acquisition and robot-assisted data collection
|   |-- dong2025_ptet_eit_reconstruction.py
|   `-- park2022_latent_projection_reconstruction.py
`-- eit_sim_to_real_multi_touch/
    |-- sim/                # Double-touch simulation and sim/real analyses
    |-- real/               # Real acquisition, calibration, and data preparation
    `-- estimation_models/  # Residual and diffusion sim-to-real models
```

### Shape reconstruction

The shape-reconstruction code supports a typical workflow of:

- generating synthetic EIT measurements and contact masks with `pyEIT`;
- reconstructing conductivity changes with methods such as back-projection (BP) and Jacobian-based reconstruction;
- training hybrid neural reconstruction models on simulated or real measurements;
- fine-tuning simulated models on real data and evaluating unseen contact shapes;
- collecting synchronized EIT/robot measurements and generating ground-truth masks.

Two standalone architecture adaptations are also provided:

- `dong2025_ptet_eit_reconstruction.py`: a masked-autoencoder/transformer PTET-style pipeline adapted to 208 voltage values and 64 x 64 contact maps.
- `park2022_latent_projection_reconstruction.py`: a three-stage voltage-autoencoder, shape-autoencoder, and latent-projection pipeline.

Both standalone scripts expect an NPZ file containing `voltages` with shape `[N, 208]` and `shapes` with shape `[N, 64, 64]` (or `[N, 1, 64, 64]`). See each script's module documentation and `--help` output for all options.

### Double-touch simulation to real

The multi-touch code simulates two circular contacts on a 16-electrode EIT domain, aligns simulated and measured channels, and learns a conditional diffusion mapping from simulated voltage changes to real-like voltage changes. The translated signals can then be used to train or evaluate a regressor for `(x1, y1, x2, y2)` contact localization.

The principal diffusion implementation is `eit_sim_to_real_multi_touch/estimation_models/002_diffusion.py`. The numbered scripts in `sim/` and `real/` reflect the experimental processing sequence; their module docstrings and command-line help document the expected inputs.

## EIT conventions

EIT injects current through boundary electrodes and measures the resulting boundary voltages. Contact changes the conductive elastomer's conductivity distribution, producing a voltage difference relative to a no-contact baseline. In this codebase:

- the sensor is generally modeled as a circular 16-electrode domain;
- `V0` denotes a no-contact baseline and `Delta V` denotes baseline-relative measurements;
- many experiments use 208 retained voltage channels under an adjacent stimulation/measurement protocol;
- simulated conductivity changes are generated with the finite-element tools in `pyEIT`;
- contact shape is represented as a 64 x 64 mask for the learned reconstruction pipelines.

Channel order, baseline convention, electrode orientation, and coordinate frame must agree between simulation and the physical sensor. The sim-to-real scripts include channel-intersection and coordinate-alignment utilities, but experiment-specific paths and calibration values should be checked before use.

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install the common dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can depend on the operating system and CUDA version. For GPU training, install the appropriate PyTorch build from the official PyTorch instructions before installing the remaining requirements.

The robot acquisition scripts additionally rely on the bundled `URBasic` code and access to a configured UR controller. `data_logger_amodo.py` also requires the lab-specific `amodo_eit` package, which is not distributed here.

## Example usage

Run commands from the repository root unless a script documents a different working directory.

Train the PTET-style shape reconstruction model:

```bash
python eit_shape_reconstruction/dong2025_ptet_eit_reconstruction.py \
  --data data/contact_shapes.npz \
  --output-dir runs/ptet
```

Train the latent-projection reconstruction model:

```bash
python eit_shape_reconstruction/park2022_latent_projection_reconstruction.py \
  --data data/contact_shapes.npz \
  --output-dir runs/park2022
```

Train the conditional diffusion sim-to-real model when both CSV files already contain baseline-relative signals:

```bash
python eit_sim_to_real_multi_touch/estimation_models/002_diffusion.py \
  --sim-csv path/to/simulated_double_touch.csv \
  --real-csv path/to/real_double_touch.csv \
  --sim-is-delta \
  --real-is-delta \
  --drop-real-zero-cols \
  --out runs/diffusion
```

Use `python <script> --help` for configurable scripts. Some exploratory scripts still contain local path constants near the top of the file; update those for your dataset before running them.

## Data and outputs

Raw/processed datasets, model checkpoints, and generated experiment outputs are intentionally excluded by `.gitignore`. To reproduce paper results, place data outside Git (or in the ignored `data/` directories) and update the relevant command-line arguments or configuration constants. Do not commit human-participant data, device identifiers, or other sensitive experimental records.

## Citation

If you use this repository, please cite the accompanying paper. A BibTeX entry will be added when the publication metadata is available.

## License

No license has been selected yet. Add a `LICENSE` file before public release to state how others may use and redistribute the code. The bundled `URBasic` source may have separate upstream licensing terms that should be verified.
