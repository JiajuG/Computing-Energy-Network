# Computing networks

Version 2026.10.27. Corrected representative-day analysis and figure source data.

## Install

Test environment: Windows 11, Python 3.9.20, 32 GB RAM, CPU only.
Use Python 3.9 with the pinned dependencies. No proprietary solver is required.
Installation normally takes about 5-15 minutes with binary packages and broadband;
this estimate is not a measured installation benchmark.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Linux and macOS have not been tested. Arial is used where available; font fallback
can alter text placement.

## Check and plot

Run from this directory. The notebooks read the supplied data directly.

```bash
python verify.py
jupyter lab
```

| File | Purpose |
|---|---|
| `figure1.ipynb` | Figure 1 and related supplementary panels |
| `figure2.ipynb` | Figure 2 and related supplementary panels, including road-distance RF |
| `figure3.ipynb` | Figure 3 and related supplementary panels |
| `bivariate.ipynb` | Bivariate SHP exports |
| `plot_utils.py` | Data loading and version checks |

Panels are written to `figures/panels`; province tiles to `exported_tiles`.
Manuscript montage assembly is manual. Stored notebook outputs and title text
are omitted. Axis labels, legends and panel identifiers are retained.
The four notebooks took about two minutes in the test environment.

RF plots use frozen predictions and drawing arrays from the accepted environment:
Python 3.9.20, NumPy 1.24.4 and scikit-learn 1.2.2. Plotting does not train a model.
`python rf.py` optionally repeats the cross-validation with the same features,
seeds and folds. RF uses road-network distance; the separate geographic
distance-bin diagnostic uses great-circle distance.

## Preprocess and solve

```bash
python preprocess.py
python run.py --list
python run.py --case local_static__S0_w0.00
```

`data/timeseries.npz` contains the corrected frozen time-series inputs.
`preprocess.py` constructs the monthly mean hourly inputs and same-day schedules.
`model.py` contains the audited SciPy/HiGHS linear programme.
`data/cases.json` contains 68 main cases, 28 sensitivities and four low-access
comparisons. `data/reference_summary.csv` contains the accepted 96-case summary.
The default command solves one case. `python run.py --all` explicitly runs the
suite sequentially and can take many hours. Existing case outputs are not overwritten.
The example above took 23 seconds and matched the reference objective.

The original 96-case production run took about 53 minutes on 36 CPU cores and
129 GB RAM, using SciPy 1.10.1 and two highspy 1.14.0 recovery solutions.
The pinned reproduction environment uses SciPy 1.13.1. Compare objectives and
constraint residuals, not bitwise-identical alternative optimal allocations.
Work files are written under `work`. Full-year validation is not included.

Load-profile sensitivity runs and their reference outputs are in
`load_profile_sensitivity/`; see its README for checks and optional reruns.

## Data and version

The figure Feathers contain the manuscript plotting values. English labels replace
the original province labels without changing numeric observations. Model and plot
inputs are checked by `VERSION.json`; the accepted result identity is recorded there.
Plots use the accepted frozen results; new model runs write separate JSON files
and do not overwrite the plotting data.
The source series are the frozen 2025 reconstruction, so no meteorological download
or new cooling simulation is needed.

## Licence

Code is released under the MIT Licence (see `LICENSE`). Processed data in `data/`
are released under CC BY 4.0; original source data remain subject to the terms of
the providers listed in Supplementary Table 1. Province boundary lines in
`boundaries/` are from the 1:4,000,000 national basic geographic dataset of the
National Geomatics Center of China.

## Repository

https://github.com/JiajuG/Computing-Energy-Network
