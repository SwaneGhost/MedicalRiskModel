# WiDS Datathon 2020 — Modeling Decisions Summary

**TL;DR:** We predict ICU mortality (`hospital_death`) with a leakage-safe pipeline in `main.ipynb`: split first, drop very sparse labs (>80% missing), add missing-indicator flags plus median imputation, train five imbalance-aware models and compare to APACHE, pick the winner by 5-fold ROC-AUC, and define intervention as flagging the **top 12%** highest-risk patients (separate from the 80% feature rule and from Kaggle probability ranking).

**Task:** Predict ICU mortality (`hospital_death`) from `training_v2.csv` (~91,713 rows, ~8.6% deaths).  
**Competition metric:** ROC-AUC on predicted probabilities.  
**Deliverable:** Leakage-safe pipeline, model comparison, metrics, and an **intervention cutoff** we can explain.

---

## What is in `main.ipynb`

| Section | Notebook content |
|--------|-------------------|
| **Setup** | Imports, constants, `WiDSFeatureBuilder`, model helpers, metrics (`build_summary_table`, `capacity_threshold`, etc.) — **helpers cell** |
| **Load & EDA** | Exploratory cells (shape, target rate, missingness, feature types). IDs stay in `df` for EDA only |
| **Preprocessing** | Split first → fit preprocessing on train → transform val/test |
| **Modeling** | 5-fold CV (ROC-AUC) → train 5 learners + APACHE baseline |
| **Cutoff** | Capacity-based threshold on validation (top 12% risk) |
| **Evaluation** | Confusion matrices, ROC, PR, calibration (incl. splines), threshold sweep, lift, decision curves, summary CSV |

**Run order:** imports → **helpers** → load → EDA → preprocess → CV → train → cutoff → plots & tables.

**Outputs:** `cv_model_comparison.csv`, `model_family_comparison_summary.csv`.

---

## What we added and changed

| Area | Before | After | Why |
|------|--------|-------|-----|
| **Preprocessing order** | Impute + one-hot on full data, then split | Split first; `WiDSFeatureBuilder` fit on **train only** | Medians and encodings must not see val/test labels or distributions (avoids optimistic metrics) |
| **Missing values** | Median fill only | Missing **flags** + median impute; drop ultra-sparse cols | EDA: “not tested” often correlates with higher risk — flags preserve that signal |
| **IDs** | Dropped at load (EDA could not inspect them) | IDs in `df` for EDA; dropped when building `X_raw` | EDA can show cardinality; models still never see IDs |
| **APACHE probs** | Partially handled | Explicitly excluded from ML features; kept as **baseline** | Same features used to build APACHE scores would let the model “cheat” vs the benchmark |
| **Models** | LR, RF, XGB, LGBM | Added **CatBoost**; class imbalance weights on all | CatBoost strong on tabular data; imbalance ~9:1 needs weighting |
| **Model choice** | Hard-coded LightGBM | **5-fold CV** on train; pick highest mean ROC-AUC | Defensible, reproducible selection aligned with Kaggle metric |
| **Cutoff** | Fixed 0.35 on a plot | **Top 12%** highest risk (capacity rule) | 0.5 is arbitrary on rare outcomes; capacity matches “how many patients can we escalate?” |
| **Metrics table** | Default 0.5 threshold for some metrics | Probabilistic metrics + operating metrics **per model at 12% capacity** | Fair comparison at same workload; APACHE at same 12%, not at 0.5 |
| **Code organization** | Scattered helpers | Single **helpers cell** (class + functions) | One place for policy constants and logic; downstream cells call them |

---

## Two different “cutoffs” (do not confuse them)

### Missing cutoff (feature-level) — **80%**

**What it is:** Drop any input **column** if more than **80%** of values are missing on the **training** split (after mapping `-1` → `NaN`).

**What it is not:** Not the intervention threshold. Not “impute vs drop” for every missing cell.

**Why 80%?**

- EDA (`summary.md`): many labs are **>50%** missing; a large block is **>70–90%** (e.g. bilirubin ~92%).
- Sparse ICU datasets often drop features with **≥80%** missing; our EDA shows the same pattern.
- Columns almost always empty add little stable signal and increase noise / overfitting risk; keeping them forces the model to learn mostly from missing flags for those labs.
- **80% on train only** so we do not use val/test missing rates (leakage).

**Trade-off:** We might drop a rare but useful lab; flags + median impute are applied to columns we **keep**.

---

### Missing-flag rule (feature-level) — **5%**

**What it is:** For numeric columns with **>5%** missing on train, add `{column}_is_missing` (1/0) **before** median imputation.

**Why?**

- EDA notes: missingness is often **informative** (test not ordered → patient may be sicker or workflow differed).
- Median imputation alone hides “was this measured?”; the flag keeps that information for tree and linear models.
- **5%** avoids thousands of flags on nearly-always-present vitals; focuses on columns where missingness is common enough to matter.

---

### Capacity cutoff (patient-level) — **12%**

**What it is:** Flag patients in the **top 12%** of predicted `P(hospital_death)` (highest risk).  
`threshold = quantile(scores, 1 − 0.12)` — each model gets its own threshold so **~12% are flagged**.

**What it is not:** Not used for Kaggle-style ranking (submissions use probabilities only). Not the 80% feature rule.

**Why 12%?**

- Death rate is ~8.6%; flagging ~12% is in the same ballpark as “review a bit more than the base rate” without flagging half the ICU.
- Defensible story for slides: *“If we can only escalate detailed review to ~1 in 8 patients, whom do we pick?”*
- Better than **0.5**: on imbalanced data, 0.5 often flags almost nobody or almost everyone; it has no link to staffing capacity.
- **Same 12% for APACHE** when comparing precision/recall — matches workload; comparing APACHE at 0.5 would be unfair.

**Test set:** We report test metrics using the same **12% rule** on test scores (one honest evaluation pass).

---

## Preprocessing pipeline (why each step)

Implemented in `WiDSFeatureBuilder` (helpers cell), used in the preprocessing cell.

1. **Drop IDs when building `X_raw`** (`encounter_id`, `patient_id`, `hospital_id`, `icu_id`)  
   **Why:** Unique or hospital-specific IDs do not generalize; they cause leakage and memorization.

2. **Drop `apache_4a_hospital_death_prob` and `apache_4a_icu_death_prob` from ML features**  
   **Why:** These are direct mortality predictions — including them makes “beating APACHE” meaningless. We still use `apache_4a_icu_death_prob` as a **baseline** on the side.

3. **Map `-1` → `NaN`**  
   **Why:** APACHE and related fields use −1 as “missing”; treating −1 as a real number distorts imputation and trees.

4. **Drop columns with >80% missing on train**  
   **Why:** See missing cutoff above.

5. **Add `_is_missing` flags (>5% missing on train)**  
   **Why:** See missing-flag rule above.

6. **Median imputation (train medians → val/test)**  
   **Why:** Simple, fast, no leakage; works with linear models after scaling. Trees also receive imputed values plus flags.

7. **One-hot encode categoricals (`drop='first'`, `handle_unknown='ignore'`)**  
   **Why:** Standard for low-cardinality strings (gender, ethnicity, APACHE body system); ignore unknown avoids crashes on rare categories in val/test.

8. **Split: 76.5% train / 13.5% validation / 10% test, stratified**  
   **Why:** Stratification preserves ~8.6% deaths in each split; held-out test touched once at the end.

---

## Models and why we included each

| Model | Why include |
|-------|-------------|
| **Logistic regression** | Interpretable linear baseline; shows whether signal is partly linear; `class_weight='balanced'` for imbalance |
| **Random forest** | Nonlinear ensemble baseline; robust default on mixed tabular features |
| **XGBoost** | Strong gradient boosting; common on Kaggle tabular tasks; `scale_pos_weight` for imbalance |
| **LightGBM** | Fast, strong on large sparse tabular data; often best out-of-the-box on this dataset type |
| **CatBoost** | Handles categoricals/missing well; `auto_class_weights='Balanced'` |
| **APACHE IVa ICU death prob** | Established clinical severity score — **must compare** to claim ML adds value |

**Not used (and why):** Deep learning (weak on this tabular set in practice), AutoML (black box, hard to explain), large stacked ensembles (out of scope for this project).

**How we pick the winner:** 5-fold **stratified CV** on the **train** split only; metric = **ROC-AUC** (competition metric). Highest mean CV AUC → `best_model_name`.

**Our CV result (example run):**

| Model | Mean ROC-AUC | Std |
|-------|--------------|-----|
| LightGBM | 0.898 | 0.004 |
| CatBoost | 0.896 | 0.004 |
| Logistic Regression | 0.884 | 0.004 |
| XGBoost | 0.879 | 0.004 |
| Random Forest | 0.877 | 0.003 |

→ **LightGBM** selected for cutoff plots and test reporting (re-run notebook if splits change slightly).

---

## Metrics (what we report and why)

### Threshold-free (ranking / Kaggle-style)

| Metric | Why |
|--------|-----|
| **ROC-AUC** | Primary competition metric; threshold-independent ranking |
| **PR-AUC (average precision)** | More informative when positives are rare (~8.6%) |
| **Brier score** | Measures calibration of predicted probabilities |
| **Lift @ top 10%** | “How many more deaths in the top decile vs random?” — useful for triage narrative |

### At capacity cutoff (~12% flagged)

| Metric | Why |
|--------|-----|
| **Sensitivity (recall)** | Of those who died, what fraction did we flag? |
| **PPV (precision)** | Of those we flagged, what fraction died? |
| **NPV** | Of those we did not flag, how safe are they? |
| **Specificity** | Of survivors, what fraction we left unflagged |
| **% flagged** | Should be ~12%; checks the policy was applied correctly |
| **F1** | Reported but secondary — unstable when positives are rare |

**Plots:** ROC, PR, calibration (reliability + splines), threshold sweep (sensitivity / PPV / % flagged), lift curves, decision curves (net benefit), confusion matrices.

---

## Scope and possible extensions

**What we implemented:** missing flags, 80% sparse drop, no APACHE prob features in ML, strong tree learners + APACHE benchmark, leakage-safe CV, explicit capacity cutoff.

**Not in this notebook (possible extensions):** large model stacks, pseudo-labeling, test-time augmentation, richer domain-specific derived features — left as future work if time allows.

---

## Artifacts

| File | Contents |
|------|----------|
| `main.ipynb` | Full pipeline: EDA, helpers, preprocess, CV, train, evaluation |
| `cv_model_comparison.csv` | 5-fold CV ROC-AUC per model |
| `model_family_comparison_summary.csv` | Validation metrics at 12% capacity + probabilistic metrics |
| `summary.md` | EDA notes (input to our thresholds) |
| `SETUP.md` | Python env / `libomp` for XGBoost & LightGBM on macOS |

---

## One-slide “why” summary

1. **Split before preprocess** — honest metrics.  
2. **80% missing cutoff** — drop useless sparse labs (per EDA).  
3. **5% missing flags** — missing often means “not tested,” which is clinical signal.  
4. **No APACHE prob in ML** — fair benchmark; compare against it instead.  
5. **Five learners + CV** — compare families; pick by ROC-AUC.  
6. **12% capacity cutoff** — defensible triage story; not arbitrary 0.5.  
7. **Rich metrics** — ranking (AUC) + operations (sens/PPV at fixed workload) + calibration.
