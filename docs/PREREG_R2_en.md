# Registration: Replication and Human-Machine Collaboration Analysis on New Datasets (R′)

> English translation prepared on 3 November 2026 for reviewers. The Chinese original is the authoritative record; for the registration files, the SHA-256 fingerprint refers to the Chinese original.

Written on: 2026-11-02. At this time WESAD, ds007554 and CogWear are being downloaded, and MAUS has not yet been obtained. **None of the data in these datasets has been looked at.** The sha256 of this file is recorded in CP9 of DECISION_LOG and will not be changed afterwards. Any deviation will be explained item by item in DECISION_LOG.

## General settings
- The processing pipeline is the same as for the original four datasets:
  - 30-second windows with a 10-second step;
  - features such as heart rate, HRV, EDA and skin temperature, z-scored against the resting segment of each person in each session;
  - within-person AUROC;
  - linear mixed-effects models with a random intercept for each person.
- When the IBI validity rate of an ECG/PPG window is below 0.8, the cardiac features are recorded as missing.
- The primary tests of each dataset are Holm-corrected within the dataset; p-values are not pooled across datasets.

## P1 MAUS: device or manipulation (corresponds to the role of ADABase in the original analysis)
- **P1a**: Trunk-ECG heart rate is higher in 3-back blocks than in 0-back blocks. Model: block mean ~ difficulty + block position + (1 | person). One-sided test of β > 0.
- **P1b**: The ratio of the wrist-PPG heart-rate effect to the ECG effect has a 95% CI lower bound > 0.5.
  - If confirmed → under a strong manipulation, the device is not the main limitation, and the negative result of UNIVERSE more likely arises from the manipulation or from the wrist EDA/skin-temperature signals themselves.
  - If the upper bound of the ratio < 0.5 → the device is the main limitation.
  - Otherwise → inconclusive; report as is.
- **P1c**: The proportion of usable windows for wrist PPG is lower than for ECG and is lower still in the hard condition (paired Wilcoxon, one-sided).
- **P1d**: Using the two blocks of the same level (0-back blocks 1 and 6; two blocks each for 2-back and 3-back), train a personal model to distinguish "different blocks of the same level". The AUROC for finger-EDA features is > 0.6 and for cardiac features is < 0.6.

## P2 WESAD: chest vs wrist, and leakage on the most commonly used benchmark
- **P2a split-half null**: Within the 20-minute baseline block, the AUROC of the personal model (k=6):
  - wrist skin temperature and EDA > 0.6;
  - chest ECG heart rate < 0.6;
  - chest skin temperature > 0.6;
  - chest EDA: no direction is specified in advance; report only.
- **P2b clock baseline**: Using only the block start time as the feature, the within-person AUROC for distinguishing stress from baseline is denoted A_clock, and is reported alongside the results of the physiological features. A_clock > 0.9 is predicted, because the baseline always comes first (the actual order of each person in the quest file is authoritative).
- **P2c**: The leave-one-subject-out (LOSO) AUROC for stress vs baseline (engagement + stress) is higher than that for stress vs amusement.

## P3 ds007554: can additional cognitive load be measured on top of movement (the core question for human-machine collaboration)
- **P3a**: ECG heart rate is higher in the "full task" (N-back mental arithmetic + active movement) than in "active movement". The block model includes repetition (1–3) and order position; one-sided test of β > 0.
- **P3b**: Active movement > passive movement (physical load, as a manipulation check).
- **P3c**: The 9-point mental-effort rating of each task is higher for the "full task" than for "active movement" (manipulation check).
- **P3d**: Block identity. There is one block for the same task in each of the three repetitions. Two evaluations of the personal model:
  - split in time within the same block;
  - across repetitions (train on repetition 1, test on repetitions 2 and 3).
  - Prediction: the former is higher than the latter (paired test, one-sided).

## P4 CogWear: across days
- **P4a**: For the personal model of Stroop vs baseline, evaluation within the same session is higher than evaluation across sessions (about two weeks later) (paired test, one-sided).
- **P4b**: Across sessions, the personal model is not better than the group model (report the difference and its 95% CI; no directional test).

## Offline decision layer for human-machine collaboration (exploratory; no significance judgements)
Using the existing estimators, simulate a policy in which "the machine adjusts its pace according to the operator's state", and compare three bases:
1. the engagement signal;
2. the graded-load signal;
3. a signal with periodic baseline refresh.

The metrics are the false-trigger rate (an adjustment is triggered when there is no change in load) and the missed-trigger rate. This part is explicitly labelled exploratory.
