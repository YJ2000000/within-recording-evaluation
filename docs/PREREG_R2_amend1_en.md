# Registration Amendment 1: SenseCobot (human-machine collaboration, written before download)

> English translation prepared on 3 November 2026 for reviewers. The Chinese original is the authoritative record; for the registration files, the SHA-256 fingerprint refers to the Chinese original.

Written on: 2026-11-02. SenseCobot (Zenodo 10124005, CC BY 4.0) has not yet been downloaded, and the data have not been looked at.

**Dataset overview** (from the Zenodo description): 21 people, a collaborative-robot programming task in three phases: introduction of learning materials, baseline measurement, hands-on practice. Signals are Shimmer3 ECG, Shimmer3 EDA, Empatica E4 (BVP, EDA, skin temperature), EEG, and NASA-TLX.

**P5 predictions** (processing pipeline as in PREREG_R2)
- **P5a engagement vs difficulty**: The within-person AUROC for programming practice vs baseline is higher than the AUROC between different practice tasks. If the practice tasks have difficulty levels, compare with the lowest level vs the highest level.
- **P5b device**: The ratio of the effect of ECG heart rate to that of E4 BVP heart rate for "practice vs baseline" has a 95% CI lower bound > 0.5. The proportion of usable windows for E4 BVP during practice is lower than for ECG.
- **P5c block identity**: Split-half null within the baseline or the longest single practice block (personal model, k=6):
  - E4 skin temperature and EDA > 0.6;
  - ECG heart rate < 0.6.
- **P5d order**: If the practice tasks are carried out in a fixed order (for example increasing difficulty), report the clock baseline, and label any "difficulty effect" as confounded with time.

**If the data structure makes any item untestable** (for example no levels, no synchronised timestamps), explain this in DECISION_LOG and report it descriptively instead.
