# WiDS ICU EDA Walkthrough

This document summarizes the analysis in [`01_wids_download_and_eda.ipynb`](01_wids_download_and_eda.ipynb). The notebook is the first exploration pass for building an ICU mortality-risk model from the WiDS Datathon 2020 Kaggle data.

## Resources

The notebook-level resources folder is [`resources/`](resources/).

Current contents:

- [`resources/APACHE_meaning.png`](resources/APACHE_meaning.png): reference image explaining APACHE concepts used in ICU severity scoring.

No additional external images need to be downloaded for the current markdown summary. However, most plots in the notebook are stored only as notebook outputs, not as image files in `resources/`. If you want those plots to appear directly inside this markdown file, export the notebook figures as PNG files into `notebooks/resources/` and reference them here.

## 1. Environment Setup

The first notebook cell handles the project setup:

- Locates the project root.
- Installs dependencies from [`../requirements.txt`](../requirements.txt).
- Imports the main EDA libraries: `pandas`, `numpy`, `matplotlib`, and `seaborn`.
- Defines `DATA_DIR` as `data/raw/widsdatathon2020`.
- Adds the project root to `sys.path` so project scripts can be imported from the notebook.

This makes the notebook mostly self-contained once the repository and requirements are present.

## 2. Kaggle Authentication

The notebook uses a local Kaggle API token file instead of typing credentials directly into cells.

Expected credential path:

```text
secrets/kaggle.json
```

The authentication cell:

- Checks that the file exists.
- Loads `username` and `key` from the JSON file.
- Validates both fields are present.
- Sets `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables for the session.

The `secrets/` directory is intentionally ignored by git, so credentials stay local.

## 3. Data Download

The notebook imports `download_wids_data` from [`../scripts/download_wids_data.py`](../scripts/download_wids_data.py).

That script:

- Downloads the WiDS Datathon 2020 files through the Kaggle API.
- Writes them to `data/raw/widsdatathon2020`.
- Checks whether the data already exists before downloading again.
- Supports credential handling through the Kaggle JSON file.

After download, the notebook lists the CSV files found in the data directory.

Expected downloaded files include:

- `training_v2.csv`
- `unlabeled.csv`
- `samplesubmission.csv`
- `solution_template.csv`
- `WiDS Datathon 2020 Dictionary.csv`

## 4. Data Overview

The notebook loads `training_v2.csv` into a dataframe called `df`.

Observed training shape:

```text
91,713 rows x 185 columns
```
with columns divided into several categories:

1. identifiers: `encounter_id`, `patient_id`, `hospital_id`, `icu_id`.
2. target: `hospital_death`.
3. APACHE-related features: columns containing `apache` in their name **
4. numeric features: continuous measurements.
5. categorical features: string or low-cardinality fields.
6. binary features: numeric fields with only two unique values.

** for apache score explanation see [APACHE_meaning.png](resources/APACHE_meaning.png)

One early note is that `gender` is stored as a string feature (categorical) with values like `M` and `F`, although it could later be encoded as a binary feature.

The target column is:

```text
hospital_death
```
The target is highly imbalanced, so future modeling should account for class imbalance through careful metrics, stratified validation, class weighting, resampling, or threshold tuning.

## 5. Missingness Overview

The notebook computes feature-level missing rates:

```python
missing = df.isna().mean().sort_values(ascending=False)
```

The first missingness table shows the features with the highest missing-value rates. The note below the table highlights that many features contain missing values, which is important because missingness may reflect either data-collection limitations or clinically meaningful absence of measurement.

### 5.1. Missingness CDF

The notebook plots a cumulative distribution of missing rates:

- Overall across all rows.
- Separately for `hospital_death = 0`.
- Separately for `hospital_death = 1`.

This plot helps identify whether there is a natural cutoff where many features move from moderately missing to heavily missing. The notebook note mentions a visible plateau that may suggest a practical threshold for removing very sparse features.

### 5.2. Most and Least Missing Features by Class

The next missingness cell compares missing-value percentages by target class.

It produces two grouped bar plots:

- Top 20 features with the most missing values.
- Top 20 features with the least missing values.

Each bar plot uses a different color for each `hospital_death` class. This helps identify features where missingness differs between survivors and non-survivors, which may become important during feature engineering.

### 5.3. Co-Missingness Heatmap

The notebook then analyzes co-missingness for the top 60 most-missing features.

For each pair of features, it calculates:

```text
fraction of rows where both features are missing
```

The heatmap displays only the upper triangle of the matrix to avoid duplicate information. The current color scale is `YlGnBu`, which keeps a blue-heavy high end while making contrast stronger than a plain blue scale.

This plot is useful for detecting groups of features that tend to be missing together. Those groups may point to shared measurement workflows, hospital-specific data availability, or related clinical panels.

## 6. Numerical Feature Distributions

The notebook summarizes numeric columns with `describe().T`, sorted by standard deviation.

This gives an initial look at:

- Feature scale.
- Spread.
- Potential outliers.
- Highly variable measurements.

This section prepares the ground for later preprocessing decisions such as scaling, clipping, transformation, and outlier handling.

## 7. Correlation Analysis

### 7.1. Correlation Heatmap

The notebook builds a numeric correlation heatmap while excluding:

- Identifier columns: `encounter_id`, `patient_id`, `hospital_id`, `icu_id`.
- Any feature whose name contains `apache`.

The goal is to inspect ordinary non-identifier, non-APACHE numeric relationships separately from ICU severity-score features. The heatmap uses Pearson correlation and centers the color scale at zero.

This step helps identify clusters of correlated continuous variables and possible multicollinearity before modeling.

### 7.2. Categorical Encoding and Hierarchical Clustering

The notebook then prepares a broader feature-clustering analysis:

- Removes identifiers.
- Removes the target.
- Excludes APACHE-related columns.
- Detects categorical columns.
- One-hot encodes categorical features using `pd.get_dummies`.
- Keeps missing category indicators with `dummy_na=True`.
- Drops constant encoded features.
- Calculates Spearman correlation across encoded and numeric features.

Before encoding, the notebook checks whether any categorical variables appear to have ordinal or incremental logic. The heuristic looks for numeric-like category values or ordered vocabularies such as low/medium/high or no/yes.

The clustering distance is:

```text
1 - absolute Spearman correlation
```

Using absolute correlation means strongly positive and strongly negative relationships are both treated as feature redundancy signals.

The notebook creates:

- A clustered Spearman correlation heatmap.
- A separate dendrogram.
- A distance-gradient coloring on the dendrogram branches.

The dendrogram gradient represents feature distance, where lower distance means stronger absolute Spearman correlation.

### 7.3. Feature Unification Cutoff Analysis

The notebook evaluates how many feature groups remain after cutting the dendrogram at different absolute Spearman correlation thresholds.

Example interpretation:

```text
cutoff = 0.90 means features are unified when |Spearman rho| >= 0.90
```

For each cutoff, the notebook records:

- Absolute Spearman cutoff.
- Equivalent dendrogram distance cutoff.
- Number of feature groups remaining.
- Number of features unified.

It then plots feature groups remaining versus the cutoff. This helps choose a redundancy threshold for future feature unification or feature removal.

### 7.4. Strong Pairwise Correlations

The notebook prints feature pairs with absolute Pearson correlation above a threshold.

Current threshold:

```text
|correlation| >= 0.8
```

This creates a direct table of highly correlated pairs, complementing the broader correlation heatmap and dendrogram.

The notebook note says that many features are strongly correlated and may cause multicollinearity.

## 8. Categorical Feature Summary

The notebook summarizes non-numeric columns by:

- Number of unique values.
- Missing rate.

This makes it easier to identify categorical variables that may need:

- One-hot encoding.
- Binary encoding.
- Grouping of rare categories.
- Missing-value handling.

## 9. Mortality Rate feature correlation

### 9.1. Numeric Features

The notebook compares numeric feature distributions against `hospital_death`.

It excludes:

- The target itself.
- Identifier columns.
- Binary-like numeric features.

For each remaining numeric feature, it plots class-separated density histograms. This helps identify measurements whose distributions differ between survivors and non-survivors.

Future notes in the notebook mention possible next steps such as combined BMI/age analysis and comparing class distributions with KDE or KL divergence.

### 9.2. Binary Features

The notebook identifies binary features as columns with exactly two non-null unique values.

It then prints the list of binary features and plots heatmaps using crosstabs against `hospital_death`.

Each binary-feature heatmap shows both:

- Raw counts.
- Percentages.

This is useful for quickly identifying binary indicators associated with different mortality rates.

### 9.3. Categorical Features

The notebook also treats low-cardinality features as categorical-like variables.

Current parameters:

```text
cat_threshold = 12
max_levels = 12
```

This means numeric columns with up to 12 unique values can be treated as categorical, while features with more than 12 levels are skipped for readability.

For each selected categorical-like feature, the notebook plots a crosstab heatmap against `hospital_death`, again showing counts and percentages.

## 10. Open Analysis Notes

The notebook currently includes TODOs for future work:

- Check which features are more missing in one class than the other.
- Explore BMI and age interactions.
- Compare class distributions with KDE and KL divergence.
- Analyze fairness or subgroup differences across ethnicity and gender.
- Write up dominant feature behavior.

These are natural next steps before model training because they connect data quality, clinical signal, and potential bias.

## Modeling Implications

The EDA points to several important modeling considerations:

- The target is imbalanced, so accuracy alone will not be sufficient.
- Missingness is widespread and may be informative.
- Some groups of features are missing together, suggesting structured missingness.
- Several features are highly correlated, so feature selection or regularized models may be useful.
- Categorical variables need careful encoding.
- APACHE-related features should be handled deliberately because they may already encode clinical severity.
- Identifier columns should not be used directly as predictive features.

This notebook is therefore a solid first pass for understanding the data before building an ICU mortality prediction pipeline.
