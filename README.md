# Hidden Quinone Discovery (CCP-tag Workflow) — Publication Package

A reproducible two-module pipeline for discovering **“hidden quinones”** in environmental char extractions using a cysteine-containing peptide (CCP) tag and LC–MS/MS data.

---

## Repository Contents

- `ccp_core.py`  
  Core scientific logic used by both modules: diagnostic fragment matching, neutral-loss and anchor checks, isotope scoring, and the rules used to make confidence calls.

- `module1_standards.py` — **Module 1: Standards (validation & calibration)**  
  Uses CCP-tagged standards to:
  - search for CCP diagnostics, neutral losses, and anchor fragments,
  - apply isotope and coverage checks,
  - summarize the evidence per candidate / cluster,
  - output a tidy table that calibrates the decision logic used later on environmental samples.

- `module2_environmental.py` — **Module 2: Environmental detection**  
  Uses environmental LC–MS/MS data to:
  - apply the same CCP diagnostics and evidence logic to unknowns,
  - aggregate evidence (diagnostics, anchors, neutral losses, isotopes),
  - assign confidence calls,
  - output tables ready for figures and supplementary material.

- `.gitignore`  
  Tells Git which files to ignore (e.g., raw MS data, large intermediate files, OS clutter).
---


