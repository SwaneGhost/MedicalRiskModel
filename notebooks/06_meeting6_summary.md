# WiDS Datathon 2020 — Meeting 6: Fairness Analysis Summary

**TL;DR:** We audited the CatBoost ICU mortality predictor (Meeting 2 best model, validation ROC-AUC = 0.893, 12 % capacity cutoff) for fairness across three protected attributes. Gender shows no meaningful disparity. Ethnicity shows modest gaps driven mainly by Hispanic over-flagging and Other/Unknown under-detection. **Age group is the most significant concern:** 45–59 year-olds have sensitivity of only 0.525, while 75+ patients are flagged at 19.3 %. The sharpest intersectional disparity is African American patients aged 16–44 (sensitivity = 0.385), visible in the full all-group heatmap. Group-specific thresholds close ethnicity gaps with < 0.07 threshold adjustments. We also implement and compare global vs group-specific isotonic recalibration: global calibration leaves the EO gap unchanged (0.085), while naive multi-calibration with a global threshold worsens it (0.358) — the correct deployment pairs group-specific calibration with group-specific thresholds.

---

## What is in `06_meeting6.ipynb`

| Section | Content |
|---------|---------|
| **Setup** | Load `eval_predictions.csv`, `X_eval_raw.csv`, `metadata.json`; build combined analysis dataframe; create age bins (16–44, 45–59, 60–74, 75+) |
| **Part 1 — Data-level fairness** | Group distributions, observed mortality rates by group, differential missingness by group |
| **Part 2 — Subgroup performance** | Per-group metrics (flagging rate, sensitivity, specificity, PPV, NPV, ROC-AUC) for gender, ethnicity, and age group; bar charts; calibration curves |
| **Part 3 — Intersectional analysis** | Age × gender and age × **all ethnicity groups** sensitivity and flagging-rate heatmaps (cells annotated with n; cells with n < 50 starred) |
| **Part 4 — Mitigation** | Group-specific threshold adjustment; before/after tables; **implemented** global and group-specific isotonic calibration with comparison; distributional justice frameworks |
| **Part 5 — Summary** | Narrative for the ICU director with actual numbers, limitations, and next steps |

**Run order:** Imports → Setup → Part 1 → Part 2 → Part 3 → Part 4 → Part 5.

**Inputs:** `best_model_info/eval_predictions.csv`, `best_model_info/X_eval_raw.csv`, `best_model_info/metadata.json` (all written by `02_meeting2.ipynb`).

---

## Baseline (overall eval set, n = 12,382)

| Metric | Value |
|--------|-------|
| Observed death rate | 8.6 % |
| Flagging rate (12 % capacity cutoff) | 12.0 % |
| Sensitivity | 0.606 |
| Specificity | 0.926 |
| PPV | 0.437 |
| NPV | 0.962 |

---

## Part 1: Data-Level Fairness

### Group sizes and observed mortality rates

| Group | N | Observed death rate |
|-------|---|---------------------|
| **Gender** | | |
| Female | 5,754 | 8.8 % |
| Male | 6,626 | 8.5 % |
| **Ethnicity** | | |
| Caucasian | 9,542 | 8.7 % |
| African American | 1,277 | 8.2 % |
| Hispanic | 490 | **9.8 %** |
| Other/Unknown | 625 | 8.6 % |
| Asian | 146 | 7.5 % |
| Native American | 117 | 6.8 % |
| **Age group** | | |
| 16–44 | 1,800 | 4.1 % |
| 45–59 | 2,786 | 5.7 % |
| 60–74 | 4,059 | 9.0 % |
| 75+ | 3,154 | **12.0 %** |

Caucasian patients dominate at 77 % of the eval set. Hispanic patients have the highest observed mortality among large ethnic groups (9.8 %). Mortality rises steeply with age: the 75+ group dies at nearly 3× the rate of the overall cohort.

### Differential missingness

Missing-value rates for key clinical features are broadly similar across gender groups. Some minority ethnic groups (African American, Hispanic) show marginally higher missingness on lab features such as `d1_lactate_max` and `d1_creatinine_max`. Because imputation replaces missing values with training-set medians (dominated by Caucasian patients), this could introduce small systematic biases for minority groups — a known limitation of median imputation in heterogeneous datasets.

---

## Part 2: Subgroup Performance at the 12 % Capacity Cutoff

### Gender

| Group | N | Deaths | AUC | Sensitivity | Specificity | PPV | Flagging Rate |
|-------|---|--------|-----|-------------|-------------|-----|---------------|
| F | 5,754 | 507 | 0.895 | **0.615** | 0.929 | 0.455 | 11.9 % |
| M | 6,626 | 562 | 0.887 | 0.598 | 0.923 | 0.420 | 12.1 % |

- **Demographic-parity gap:** 0.002 (negligible)
- **Equal-opportunity gap:** 0.017 (negligible)

The model is effectively gender-fair. Female patients have marginally higher sensitivity and PPV. No corrective action is needed.

---

### Ethnicity

| Group | N | Deaths | AUC | Sensitivity | Specificity | PPV | Flagging Rate |
|-------|---|--------|-----|-------------|-------------|-----|---------------|
| African American | 1,277 | 105 | 0.891 | 0.619 | 0.943 | 0.492 | 10.3 % |
| Asian | 146 | 11 | 0.952* | 0.727* | 0.948 | 0.533 | 10.3 % |
| Caucasian | 9,542 | 831 | 0.889 | 0.600 | 0.924 | 0.429 | 12.2 % |
| Hispanic | 490 | 48 | 0.901 | 0.646 | 0.916 | 0.456 | 13.9 % |
| Native American | 117 | 8 | 0.911* | 0.750* | 0.945 | 0.500 | 10.3 % |
| Other/Unknown | 625 | 54 | 0.894 | 0.593 | 0.909 | 0.381 | 13.4 % |

*\* Unreliable: < 15 deaths; one mis-classified death shifts sensitivity by ~0.12.*

- **Demographic-parity gap (all groups):** 0.036
- **Equal-opportunity gap (all groups):** 0.157 (driven by unreliable small groups; **~0.053 among reliable groups only**)

Key observations among reliable groups:
- **Hispanic** is over-flagged (13.9 %) and over-sensitive (0.646), tracking their higher observed mortality. Group-specific threshold correction reduces flagging to 11.0 %.
- **African American** is under-flagged (10.3 %) but achieves above-average sensitivity (0.619). The model is more conservative but not less accurate for this group.
- **Caucasian** sits close to the overall (sens = 0.600, flag = 12.2 %).
- **Other/Unknown** has the worst trade-off: lowest sensitivity (0.593) with above-average flagging (13.4 %).

---

### Age Group

| Group | N | Deaths | AUC | Sensitivity | Specificity | PPV | Flagging Rate |
|-------|---|--------|-----|-------------|-------------|-----|---------------|
| 16–44 | 1,800 | 74 | **0.944** | 0.581 | 0.977 | 0.518 | 4.6 % |
| 45–59 | 2,786 | 158 | 0.911 | **0.525** | 0.962 | 0.451 | 6.6 % |
| 60–74 | 4,059 | 365 | 0.880 | 0.578 | 0.921 | 0.420 | 12.4 % |
| 75+ | 3,154 | 379 | 0.857 | **0.673** | 0.873 | 0.419 | 19.3 % |

- **Demographic-parity gap:** 0.147
- **Equal-opportunity gap:** 0.148

Three distinct patterns:

1. **16–44 — best discrimination but under-flagged by design.** AUC = 0.944 is the highest of any subgroup, meaning the model ranks young high-risk patients very well. However, the global 12 % threshold calibrated to the full population only fires for 4.6 % of young patients (base rate 4.1 %). The model is not wrong; the threshold is population-level, not age-specific.

2. **45–59 — most under-served group.** Sensitivity = 0.525 is the lowest in the audit. Fewer than 1 in 2 middle-aged patients who die is flagged. The cause is not poor discrimination (AUC = 0.911) but that this group sits in a risk zone where the global threshold is ill-positioned.

3. **75+ — appropriately flagged most aggressively.** Flagging rate of 19.3 % tracks the 12.0 % observed mortality. The large flagging-rate gap (14.7 pp) is not unfair — it reflects the model correctly prioritising the highest-risk group.

---

### Calibration

- **Gender:** Both curves track closely and lie near the diagonal — no systematic mis-calibration by gender.
- **Ethnicity:** Minor differences visible in the mid-probability range; no severe over- or under-estimation for large groups.
- **Age group:** 16–44 and 45–59 show slight under-estimation in the moderate-risk zone; 75+ is reasonably well-calibrated. Full within-group recalibration (multi-calibration) is a recommended next step.

---

## Part 3: Intersectional Analysis

### Age × Gender (sensitivity)

| Age group | F | M |
|-----------|---|---|
| 16–44 | 0.545 | 0.610 |
| 45–59 | 0.545 | 0.511 |
| 60–74 | 0.557 | 0.597 |
| 75+ | 0.676 | 0.670 |

Gaps are modest (≤ 0.065). The largest: 16–44 male (0.610) vs female (0.545) — young women who die are somewhat less likely to be flagged than young men. By age 75+ the gap is negligible.

### Age × Ethnicity (all groups, sensitivity)

The analysis now shows all ethnic groups (min 15 patients per cell; cells with n < 50 starred). Asian and Native American patients do not appear because they have fewer than 5 deaths per age-group cell — a hard limit for classification metrics.

| Age group | African American | Caucasian | Hispanic | Other/Unknown |
|-----------|-----------------|-----------|----------|---------------|
| 16–44 | **0.385** (n=280) | 0.653 (n=1,182) | 0.400† (n=86) | — |
| 45–59 | 0.571 (n=374) | 0.500 (n=2,065) | 0.400† (n=95) | 0.692 (n=138) |
| 60–74 | 0.538 (n=409) | 0.594 (n=3,193) | 0.400 (n=132) | 0.476 (n=191) |
| 75+ | **0.833** (n=169) | 0.645 (n=2,628) | **0.826** (n=146) | 0.800 (n=123) |

*† Hispanic 16–44 and 45–59 cells each have an estimated < 10 deaths despite n > 50 (low base-rate age band × small ethnic group) — sensitivity of 0.400 should be treated as a rough estimate only.*

Key observations from the full view:
- **African American 16–44 (0.385)** remains the sharpest disparity — a 26.8 pp gap vs Caucasian 16–44 (0.653).
- **Hispanic patients** show consistent patterns: lower sensitivity in younger/mid age bands (0.400) but very high sensitivity at 75+ (0.826), tracking their elevated mortality in that cohort.
- **African American and Hispanic 75+** both substantially exceed Caucasian 75+ (0.833 and 0.826 vs 0.645) — the model is more sensitive for older minority patients.
- **Other/Unknown 45–59** shows unexpectedly high sensitivity (0.692) but is a heterogeneous category; interpretation is limited.

The disparity for African American patients is therefore not uniform across age — it is concentrated at 16–44 and reverses at 75+.

---

## Part 4: Mitigation — Group-Specific Threshold Adjustment

### Method

For each subgroup we find the threshold $t_g$ that minimises $|\text{sensitivity}_g(t_g) - 0.606|$. All adjustments are post-hoc (no retraining required).

### Gender results

| Group | Old threshold | Old sens | Old flag | New threshold | New sens | New flag |
|-------|--------------|----------|----------|--------------|----------|----------|
| F | 0.6079 | 0.615 | 11.9 % | 0.6132 | 0.606 | 11.6 % |
| M | 0.6079 | 0.598 | 12.1 % | 0.5928 | 0.607 | 12.7 % |

Adjustments are negligible. Not worth deploying separately.

### Ethnicity results

| Group | Old threshold | Old sens | Old flag | New threshold | New sens | New flag |
|-------|--------------|----------|----------|--------------|----------|----------|
| African American | 0.6079 | 0.619 | 10.3 % | 0.6132 | 0.610 | 10.0 % |
| Caucasian | 0.6079 | 0.600 | 12.2 % | 0.6001 | 0.606 | 12.6 % |
| Hispanic | 0.6079 | 0.646 | 13.9 % | **0.6691** | 0.604 | **11.0 %** |
| Other/Unknown | 0.6079 | 0.593 | 13.4 % | 0.5991 | 0.611 | 13.6 % |

The most impactful correction is for **Hispanic patients**: raising the threshold from 0.608 to 0.669 removes 2.9 pp of over-flagging (13.9 % → 11.0 %) while still meeting the sensitivity target — a net benefit with no accuracy cost.

For **Other/Unknown**, a small threshold decrease (0.608 → 0.599) closes the sensitivity shortfall from 0.593 to 0.611.

All four adjustments are < 0.07 threshold units — operationally trivial.

### 4.1 Calibration: Global vs Multi-Calibration

We implemented two post-hoc isotonic recalibration strategies and evaluated them on a held-out half of the eval set (n ≈ 6,191, stratified split).

| Strategy | EO gap (sensitivity) | Dem-parity gap (flagging) |
|----------|---------------------|--------------------------|
| Uncalibrated | 0.085 | 0.045 |
| Global isotonic | 0.085 | 0.042 |
| Multi-calibration (group-specific isotonic) | **0.358** | **0.155** |

**Global isotonic calibration** reshapes the overall probability scale but barely moves the EO gap (0.085 → 0.085). The relative ranking across groups is almost unaffected.

**Naive multi-calibration worsens the EO gap (0.085 → 0.358).** Group-specific isotonic regression rescales each group's probabilities independently, which distorts cross-group rankings. When a single global 12 % threshold is then applied to the rescaled scores, groups whose probabilities were compressed (e.g. African American) fall below the cutoff while groups whose scores were inflated (e.g. Other/Unknown) rise above it — producing large sensitivity swings in opposite directions.

**The fix:** multi-calibration must be paired with group-specific thresholds to work as intended. The correct deployment sequence is:

1. Fit per-group isotonic calibrators (correct within-group probability bias).
2. Derive per-group thresholds from the recalibrated scores (equalise sensitivity).

This is the pipeline described in Barda et al. JAMIA 2021. Combining both steps corrects probability mis-calibration and equal-opportunity gaps simultaneously without retraining.

**Multi-calibration** (Hébert-Johnson et al. 2018) formally requires that for every subgroup $G$ and predicted-probability level $v$:

$$\mathbb{E}[Y \mid \hat{p}(X) = v,\; X \in G] = v$$

This is stronger than group-level calibration — it must hold simultaneously for all groups and all risk strata.

### 4.2 Distributional Justice Frameworks

| Framework | Principle | Implication |
|-----------|-----------|-------------|
| **Utilitarianism** | Maximise aggregate lives saved | Single threshold maximises overall sensitivity; may harm small minority groups |
| **Prioritarianism** | Extra weight to the worst-off | Equal sensitivity protects all at-risk patients equally, regardless of group |
| **Desert** | Benefit proportional to clinical need | Remove demographic bias; do not create new privilege |

**Our recommendation:** Equal-opportunity (equal sensitivity) adjustment is consistent with prioritarianism and clinical ethics. Group-specific thresholds for ethnicity are recommended. Age-specific thresholds for the 45–59 group require further investigation (the issue is base-rate positioning, not model discrimination).

---

## Part 5: Summary for the ICU Director

### Gender — no action needed
Female and male patients receive effectively identical model treatment. The 1.7 pp sensitivity gap (0.615 vs 0.598) is within noise. No threshold adjustment is warranted.

### Ethnicity — targeted threshold adjustment recommended
- Deploy a raised threshold (0.6691) for **Hispanic patients** to reduce over-flagging from 13.9 % to 11.0 %. Sensitivity remains at target.
- Deploy a slightly lowered threshold (0.5991) for **Other/Unknown patients** to close the sensitivity gap (0.593 → 0.611).
- African American and Caucasian thresholds require only negligible changes.
- Do not rely on model output for **Asian or Native American patients** until more data is collected (< 150 patients, < 12 deaths each).

### Age — the most urgent clinical concern
**45–59 year-olds have sensitivity of only 0.525** at the shared threshold — the worst of any age group and 14.8 pp below the 75+ group (0.673). This is not a discrimination problem (AUC = 0.911) but a threshold-positioning problem: the global 12 % rule does not adequately cover a group with 5.7 % base mortality. Investigate a lower age-specific sub-threshold or age-stratified flagging capacity for this cohort.

### Intersectional — targeted monitoring
**African American patients aged 16–44** have sensitivity = 0.385, a 26.8 pp gap vs Caucasian peers. This is the highest-priority monitoring cell. We recommend flagging this population for prospective outcome tracking from day one of deployment and collecting additional data before extending the model to facilities with high proportions of young Black patients.

---

## Limitations

| Limitation | Impact |
| ---------- | ------ |
| Eval set only (12,382 patients) | Subgroup CIs are wide for small ethnic groups; validate thresholds on held-out test set |
| `hospital_id` not available | Hospital-level disparities (staffing, equipment, patient mix) cannot be audited |
| Asian / Native American too small for intersectional | Fewer than 5 deaths per age-group cell; excluded from heatmap |
| Multi-calibration evaluated on 50 % of eval set | Calibration results are indicative; should be validated on the full test set |
| Historical bias in training data | Model inherits bias from past clinical decisions (test ordering, ICU admission) |
| Age demographic-parity gap partly normative | Imposing strict demographic parity on age would harm high-risk elderly patients |

## Recommended Next Steps

1. Validate ethnicity-specific thresholds (Hispanic: 0.6691, Other/Unknown: 0.5991) on the held-out test set.
2. Deploy the correct multi-calibration pipeline: group-specific isotonic recalibration **followed by** group-specific thresholds on the recalibrated scores.
3. Investigate age-specific sub-thresholds for the 45–59 cohort to raise sensitivity from 0.525.
4. Flag the African American × 16–44 intersectional cell for prospective post-deployment monitoring; collect more data before deploying in facilities with high proportions of young Black patients.
5. Re-audit with `hospital_id` if the preprocessing pipeline is revised.

---

## Artifacts

| File | Contents |
|------|----------|
| `06_meeting6.ipynb` | Full fairness analysis notebook (39 cells) |
| `best_model_info/eval_predictions.csv` | Model predictions on eval set (input) |
| `best_model_info/X_eval_raw.csv` | Raw eval features with protected attributes (input) |
| `best_model_info/metadata.json` | Model metadata, thresholds, overall metrics (input) |
