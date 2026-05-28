# WiDS Datathon 2020 — Meeting 3: Model Explainability Summary

**TL;DR:** We explained the best model from Meeting 2 (**CatBoost**, test ROC-AUC = 0.907) using three global methods (SHAP, permutation importance, PDP, LASSO LR) and three local methods (SHAP waterfall, LIME, counterfactual), then compared the model's top features against the APACHE IVa clinical score. The model largely validates known clinical signals while adding first-day vital-sign measurements and missingness flags that APACHE does not use.

**Addressed questions from the ICU director:**
1. **Global explainability** — which features drive the model overall?
2. **Local explainability** — why did the model flag *this specific patient*?
3. **APACHE comparison** — are the model's signals consistent with established clinical knowledge?

---

## What is in `03_meeting3.ipynb`

| Section | Content |
|---------|---------|
| **Setup** | Imports, `WiDSFeatureBuilder` redefinition (required for joblib deserialization), constants |
| **Load artifacts** | Load `feature_builder.joblib`, `best_model.joblib`, processed eval/test sets, raw eval data (for LIME), APACHE scores, and `metadata.json` from `best_model_info/` |
| **Part 1 — Global** | SHAP bar + beeswarm, SHAP dependence plots (top 4 features), permutation feature importance, PDPs, LASSO LR odds ratios |
| **Part 2 — Local** | Select 3 patients (high risk, low risk, borderline), SHAP waterfall plots, LIME surrogate, counterfactual search |
| **Part 3 — APACHE** | Map APACHE IVa known variables to dataset columns; compare with top 30 SHAP features; side-by-side importance bar chart |

**Run order:** imports → WiDSFeatureBuilder definition → load artifacts → global explainability cells → patient selection → local explainability cells → APACHE comparison.

**Inputs:** artifacts from `best_model_info/` (written by `02_meeting2.ipynb`).

---

## Loaded artifacts

| File | Contents |
|------|----------|
| `best_model_info/feature_builder.joblib` | Fitted `WiDSFeatureBuilder` (preprocessing pipeline) |
| `best_model_info/best_model.joblib` | Fitted CatBoost sklearn Pipeline |
| `best_model_info/X_eval_processed.csv` | Preprocessed validation set (12,382 × 277) |
| `best_model_info/X_eval_raw.csv` | Raw validation set (for LIME) |
| `best_model_info/X_test_processed.csv` | Preprocessed test set (9,172 × 277) |
| `best_model_info/metadata.json` | `best_model_name`, `validation_threshold_top_12pct` (= 0.6075) |

---

## Part 1: Global Explainability

Global methods describe how features affect predictions **on average across all patients**.

### 1.1 SHAP (TreeSHAP)

SHAP (SHapley Additive exPlanations) assigns each feature a contribution for each prediction, derived from cooperative game theory.

| Plot | What it shows |
|------|---------------|
| **Bar plot** | Mean \|SHAP\| per feature — global feature importance ranking |
| **Beeswarm** | One dot per patient; x-axis = SHAP value, color = feature value (red=high, blue=low) — shows direction and distribution |
| **Dependence plots** | SHAP value vs. feature value for top 4 features — reveals non-linearity and interactions |

**Why TreeSHAP:** exact (not approximate) for tree models; fast even on 12k patients × 277 features.

**How to read the beeswarm:** red dots on the right = high feature values increase risk (e.g., high age). Blue dots on the right = low feature values increase risk (e.g., low GCS).

---

### 1.2 Permutation Feature Importance

Measures the drop in **ROC-AUC** when a feature's values are randomly shuffled, breaking its relationship with the outcome.

**Advantage over SHAP:** computationally independent; reflects actual model *error degradation*.  
**Limitation:** can underestimate correlated features — shuffling one may still leave signal in a correlated feature.

---

### 1.3 Partial Dependence Plots (PDP)

Shows the **average predicted mortality risk** as a function of one feature, marginalizing over all other features.

- Applied to the top numeric features from SHAP (binary flags and `_is_missing` indicators excluded since they have only 2 values).
- Most relationships are **monotonic and clinically plausible** (e.g., risk rises with age, falls with GCS).

**Limitation:** assumes feature independence; rare or impossible patient profiles may appear in some regions when features are correlated.

---

### 1.4 LASSO Logistic Regression: Odds Ratios

Logistic regression with **L1 penalty** (LASSO) to obtain a sparse, fully interpretable model.

| Parameter | Value |
|-----------|-------|
| Penalty | L1 (`liblinear` solver) |
| C (inverse regularization) | 0.1 (strong sparsity) |
| Class weight | `balanced` |

**Why include it:** LR was not the best model (AUC ~0.882 vs CatBoost 0.907), but it provides odds ratios that clinicians can audit without understanding trees. Useful for regulatory review and slide decks.

**Odds ratio interpretation:** OR > 1 = increases odds of death; OR < 1 = decreases odds of death.

---

## Part 2: Local Explainability

Local methods explain **why a single patient received their specific risk score**. Three patients were selected:

| Patient | Selection rule | Predicted P(death) | Actual outcome |
|---------|---------------|-------------------|----------------|
| **A** | Highest risk among true deaths flagged above threshold | 1.000 | Died |
| **B** | Lowest risk among true survivors below threshold | 0.000 | Survived |
| **C** | Closest to the 12% threshold (= 0.6075) | 0.607 | Survived |

### 2.1 SHAP Waterfall Plots

Shows how each feature shifts the prediction from the **population baseline** to the final risk for one patient. Red bars push risk up; blue bars push risk down. Produced for all three patients.

### 2.2 LIME

LIME (Local Interpretable Model-agnostic Explanations) fits a **local linear model** around the patient being explained by perturbing input features and re-querying the black-box model.

**Advantage:** produces short, selective, human-friendly explanations without needing to understand the underlying model.  
**Limitation:** the local neighborhood size is a tunable hyperparameter; results can vary slightly between runs.

Applied to Patients A and B (extreme cases are more instructive than the borderline patient for this method).

### 2.3 Counterfactual Explanation

Answers: *"What is the minimum change to this patient's features that would move them below the 12% threshold?"*

**Implementation:** gradient-free optimization (`Nelder-Mead`) minimizes squared distance to `threshold − 0.05`, with L2 regularization to prefer small perturbations, over the top 20 actionable numeric features (missing flags excluded).

**Result for Patient C (marginally-flagged):** the optimizer found no significant feature changes were needed — the patient's risk (0.607) was already exactly at the threshold (0.6075). No clinically interesting counterfactual was produced; the patient was already on the decision boundary.

**Takeaway:** counterfactuals are most informative for patients who are *just above* the threshold with some slack, not for patients sitting exactly on the boundary.

---

## Part 3: APACHE Comparison

### APACHE IVa known variables (mapped to dataset columns)

APACHE IVa uses: age, heart rate, MAP, temperature, respiratory rate, 24h urine output, PaO2/FiO2, creatinine, bilirubin, sodium, glucose, hematocrit, WBC, GCS (unable), intubation/ventilation, ARF, elective surgery, chronic conditions (AIDS, hepatic failure, lymphoma, metastatic cancer, immunosuppression, cirrhosis, diabetes), and admission diagnosis.

### Overlap between top 30 CatBoost features (by mean |SHAP|) and APACHE

| Category | Count | Examples |
|----------|-------|---------|
| **Shared** (in both) | 3 | `age`, `urineoutput_apache`, `ventilated_apache` |
| **Model-only** (in top 30, not APACHE) | 27 | `d1_*` first-day measurements, `gcs_eyes/motor/verbal_apache`, `bmi`, `pre_icu_los_days`, `d1_lactate_min`, `icu_admit_source_*` |
| **APACHE-only** (in APACHE, not top 30) | 23 | `creatinine_apache`, `map_apache`, `temp_apache`, `wbc_apache`, `hematocrit_apache`, chronic comorbidities |

### Key takeaways for the ICU director

1. **The model validates clinical intuition** — APACHE variables that appear in the top features (age, urine output, ventilation) carry meaningful SHAP values, confirming the model has learned established clinical signals.

2. **The model captures additional signal** — first-day measurements (`d1_*`: vital signs, lactate, BUN, hemoglobin, SpO2) and GCS sub-scores (`gcs_eyes_apache`, `gcs_motor_apache`, `gcs_verbal_apache`) contribute beyond the single-value APACHE summaries. This explains the AUC gap (0.907 vs 0.837).

3. **Missingness flags are informative** — features like `d1_arterial_pco2_max_is_missing` carry SHAP weight because *not ordering a test is itself a clinical signal* (e.g., patient too unstable, or clinician judged the test unnecessary).

4. **No unexpected or suspicious features** — the model does not rely on patient identifiers, random correlates, or clinically implausible drivers.

---

## Three global methods compared

| Method | What it measures | Advantages | Limitations |
|--------|-----------------|------------|-------------|
| **SHAP** | Contribution of each feature to each prediction | Theoretically grounded; shows direction; exact for trees | Computationally heavy for large samples (sampled here) |
| **Permutation importance** | Drop in ROC-AUC when feature is shuffled | Independent of model internals; reflects actual error | Underestimates correlated features |
| **PDP** | Average predicted risk vs. feature value | Reveals non-linearity and shape of effect | Assumes feature independence |
| **LASSO LR** | Linear odds ratios | Fully transparent; auditable by clinicians | Lower AUC; cannot capture interactions |

---

## Two local methods compared

| Method | Approach | Advantage | Limitation |
|--------|----------|-----------|------------|
| **SHAP waterfall** | Exact additive feature contributions | Consistent with global SHAP; exact for trees | Requires understanding SHAP framing |
| **LIME** | Local linear surrogate | Short, human-readable explanation | Neighborhood size is a tuning parameter; can be unstable |
| **Counterfactual** | Minimum feature change to cross threshold | Actionable ("what would change the decision?") | Requires a meaningful margin from the threshold; Patient C sat exactly on the boundary |

---

## Artifacts

| File | Contents |
|------|----------|
| `03_meeting3.ipynb` | Full explainability notebook |
| `best_model_info/` | All artifacts loaded from Meeting 2 (model, data, metadata) |
