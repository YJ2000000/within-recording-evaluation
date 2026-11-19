# Within-recording evaluation of personalised wearable workload and stress models

Code and results for the manuscript

> Jin, Y.; Chen, M. *Within-Recording Evaluation Inflates Personalised Wearable Detection of Workload and Stress: Block Identity, Drift and Simple Controls.* Submitted to *Sensors* (MDPI), 2026.

The study re-analyses seven public datasets (COG-BCI, UNIVERSE, MultiPhysio-HRC, WESAD, MAUS, ds007554, CogWear) with one pipeline and asks how much of the accuracy of personal workload/stress models evaluated within one recording reflects load rather than block identity and drift.

## Contents

| Folder | What it holds |
|---|---|
| `pipeline/` | Download helpers for the public datasets (`dl_*.py`, `get_cogload.sh`), ECG quality check and inventory scripts |
| `pipeline/xtask/` | Windowing, features and all analyses (see the map below) |
| `paper/` | Scripts that draw the figures (`make_figures.py`, `extra_for_figs.py`) and build the Supplementary Materials (`make_supplement.py`); `data/` holds the values plotted in each figure and the table contents; `refs/` the reference metadata |
| `results/` | All result tables (CSV/JSON) produced by the pipeline and used in the paper |
| `docs/` | The files of claims written down before the held-out analyses |

The datasets themselves and the derived window files are **not** redistributed here; download them from their original sources (links in the paper's Data Availability Statement) under their own licences.

## Where each result comes from

| Paper | Script (stage) | Result files |
|---|---|---|
| Windows, beat detection, features (Section 3.2) | `xtask/build_windows.py`, `xtask/build_windows2.py`, `xtask/features.py`, `xtask/wesad_compact.py`, `ecg_qc.py` | window files (not included) |
| Difficulty and drift effects (Section 4.1, Figure 2, Table 3) | `xtask/confirm.py`, `xtask/r3_analyses.py` (`bpm`, `bracket`, `r5engage`), `xtask/r3_stats.py` | `results/confirm/`, `results/r3/bpm.csv`, `robustness.csv`, `W_confirm_r3.csv` |
| Split-half null, gaps, random forests (Section 4.2, Figure 3a,b; Tables S2, S7) | `xtask/r3_analyses.py` (`gapnull`, `rfnull`) | `results/r3/gapnull_*` |
| Shared-direction test of the null (Section 4.2, Figure 3a; Table S9) | `xtask/r6_drift_test.py` (`sim`, `data`), `xtask/r7_analyses.py` (`power`) | `results/r3/r6_*`, `r7_power*` |
| Same-condition two-block control (Section 4.3, Figure 3c; Table S8) | `xtask/prereg_r2.py` (MAUS), `xtask/r3_analyses.py` (`samelevel`, `r5samelevel`) | `results/prereg_r2/`, `results/r3/samelevel_*`, `r5_samelevel_*` |
| Protocols, clock, calibration budget, score–time test, MAUS cross-block (Section 4.4, Figure 4; Table S10) | `xtask/clock_and_degrade.py`, `xtask/r3_analyses.py` (`gapreal`, `crossblk`), `xtask/r7_analyses.py` (`protocols`, `maus`, `cogwear`) | `results/confirm/clock_baseline.csv`, `results/r3/gapreal_*`, `r7_*` |
| Bootstrap intervals for heart-rate effects (Table S6) | `xtask/r7_analyses.py` (`bootci`) | `results/r3/r7_bpm_bootci.csv` |
| MAUS position check, missingness-only model (Sections 4.1 and 4.5; Table S11) | `xtask/r9_analyses.py` (`mauscheck`, `missing`) | `results/r3/r9_*` |
| Signal quality and agreement (Section 4.5, Figure 5; Tables S4, S5) | `xtask/r3_analyses.py` (`usable`, `agree`) | `results/r3/usable.csv`, `agree.csv` |
| Held-out claims and verdicts (Section 4.6, Table 4; Table S1) | `xtask/prereg_r2.py`, `xtask/r3_verdicts.py` | `results/prereg_r2/`, `results/r3/prereg_verdicts.csv` |

Run a stage with, for example:

```bash
export DATA_ROOT=/path/to/data/raw          # raw downloads; windows go to ../windows, results to ../results
export WIN_DIR=/path/to/data/windows
cd pipeline/xtask
python r3_analyses.py gapnull
python r7_analyses.py protocols maus
```

Default paths in the scripts point to `/path/to/project/`; set the environment variables above or replace this prefix with your own project folder before running.

## Claims written down before the held-out analyses

`docs/PREREG_R2_en.md` and `docs/PREREG_R2_amend1_en.md` are English translations of the claims files. The originals, `docs/PREREG_R2.md` and `docs/PREREG_R2_amend1.md`, were written in Chinese and are kept unchanged only so that their SHA-256 fingerprints, given in the paper, can be checked:

```
PREREG_R2.md         1a37c84feadf629eddda27737db3d3a0183483a1c6ae6ab26b0ba0ac806fa73c
PREREG_R2_amend1.md  9afe86e339de4deb17c98f7bc29562232ef7b52673d466168f54028539914d84
```

The files were registered internally and are not externally time-stamped; all deviations are listed in Supplementary Table S1.

## Software

Python 3.13 with the packages in `requirements.txt`.

## Licence

Code: MIT licence (`LICENSE`). Result tables and documents: CC BY 4.0.
