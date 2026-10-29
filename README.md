# EPFL Machine Learning (CS-433) – Project 1: Predicting Heart Disease Risk (BRFSS 2015)

**Team:** `Milos Tomic`, `Ismail Temmar`, `Darinka Loncar`  
**AIcrowd team names:** `losmi (Milos Tomic), darinka (Darinka Loncar), Rammet (Ismail ), best submission: 337781`  
**Course:** Machine Learning CS-433, EPFL, Fall 2026

Binary classification of whether a respondent has coronary heart disease / myocardial infarction (MICHD), using the BRFSS 2015 telephone-survey data (> 300,000 individuals, 321 raw features). Predictions are submitted as `+1` (MICHD) / `-1` (no MICHD).

**Constraints respected:** only Python standard library, NumPy and (for plots only) matplotlib. No Pandas, scikit-learn, PyTorch, etc. No external data or code.

---

## 1. Repository structure

| Path | Description |
|---|---|
| `implementations.py` | The 6 required ML methods (see Section 3), plus small shared helpers/constants. |
| `run.py` | **Reproduces our best submission** end-to-end from the raw CSVs (see Section 2). |
| `process.py` | Fast CSV loading (`read_csv`) and joining of the labels to the feature rows (`join_targets`). |
| `recode_brfss.py` | All data cleaning and feature engineering (see Section 4). |
| `codebook_values.json` | BRFSS 2015 codebook (value labels per column) that drives the recoding. |
| `gbdt.py` | Our own NumPy implementation of histogram-based gradient-boosted decision trees (see Section 5). |
| `helpers.py` | Course-provided helpers (`load_csv_data`, `create_csv_submission`, ...). |
| `grading_tests/` | Course-provided tests for the 6 required functions. |
| `report/` | 2-page LaTeX report (source and PDF). |
| `latex-example-paper/` | Course-provided LaTeX template. |
| `project1_description.pdf` | Project description. |

---

## 2. How to reproduce our best submission

### Requirements
- Python 3.9+
- `numpy`
- `matplotlib` (only needed for figures, not for `run.py`)

```bash
pip install numpy matplotlib
```

### Data
Download the files from the [AIcrowd challenge page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) and place them, **with their original names**, in a folder `dataset/` at the repository root:

```
dataset/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

The data is not stored in this repository.

### Run
```bash
python run.py                 # writes submission.csv
python run.py my_output.csv   # or choose the output file name
```

`run.py` runs the full pipeline from the raw CSV files, with no cached intermediate files:

1. Read `x_train.csv`, `x_test.csv`, `y_train.csv`.
2. `preprocess()`: recoding driven by `codebook_values.json`, skip-logic fills, dropping of columns. Missing values are kept as `NaN` (they get their own bin in the trees).
3. `fit_binner()`: per-feature histogram bins, fitted on the training set only.
4. `fit_gbdt()`: 500 trees on the full training set, fixed seed (`seed=0`).
5. Predict probabilities, apply the decision threshold **0.21**, apply the `HAREHAB1` override (Section 4.6), and write the CSV with `create_csv_submission`.

The script is deterministic (seeded) and produces exactly the CSV we submitted for our best score. It performs training and inference, so no pretrained weights are needed.

---

## 3. Required method implementations (`implementations.py`)

All functions return `(w, loss)`: the **last** weight vector and its loss. All vectors are 1D arrays of shape `(D,)`. The loss returned by the regularized methods does **not** include the penalty term.

| Function | Method |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression, gradient descent (MSE with factor ½) |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression, SGD with mini-batch size 1 |
| `least_squares(y, tx)` | Least squares via normal equations (no `numpy.linalg.lstsq`) |
| `ridge_regression(y, tx, lambda_)` | Ridge regression via normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression, gradient descent, `y ∈ {0, 1}` |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | L2-regularized logistic regression, penalty `λ‖w‖²`, `y ∈ {0, 1}` |

To check them with the course tests (from the repository root):

```bash
pip install pytest
pytest grading_tests
```

(`pytest` is only used for testing, not in our code.)

---

## 4. Data preparation and feature engineering (`recode_brfss.py`)

The raw data are survey codes (e.g. "7 = don't know", "888 = none", `1xx` = per week, `2xx` = per month), so the raw numbers cannot be fed to a model directly. Everything below is documented in the module docstring.

### 4.1 Recoding (`recode`), driven by `codebook_values.json`
1. "Don't know / not sure / refused / missing" codes → `NaN`.
2. Packed encodings decoded to natural units: alcohol days, exercise frequency, fruit/vegetable intake, blood-sugar checks (→ per day/week/year), exercise duration (→ minutes), weight (→ kg), height (→ cm), dates (→ months before the interview).
3. Ordinal scales whose "never/none" code is out of order are fixed (`ORDINAL_FIXES`).
4. "None/Zero/Never" codes of count variables → `0`.
5. Yes/No variables → `1`/`0`.

### 4.2 Skip-logic fills (`fill_skip_logic`)
A follow-up question that was skipped because of the answer to its filter question receives the value that answer implies (e.g. `BPMEDS = 0` if the respondent has no high blood pressure; non-diabetics get `0` for all diabetes-module questions).

### 4.3 Engineered features
- `UNSURE_CHRONIC`: 1 if the respondent answered "don't know" to any chronic-condition question.
- `CDSCORE`: combined cognitive-decline score (from `CDHOUSE`, `CDASSIST`, `CDSOCIAL`).
- `DIABYRS`: years with diabetes (from age at diagnosis and current age).
- `HIV5YR`: HIV test within the last 5 years.
- `NUMADULT` merged with `HHADULT` (same question asked on cell vs. landline questionnaires).
- Codes not ordered by severity are re-ordered (`BPHIGH4`, `DIABETE3`, `PERSDOC2`); pre-diabetes is folded into `DIABETE3` as "borderline".

### 4.4 Dropped columns
- Interview bookkeeping, survey weights, phone-screening questions, state-specific optional modules with no filter question, child-related columns, CDC missingness flags (`DROP_COLUMNS`).
- Columns whose value is fully determined by a kept column (raw answers replaced by CDC calculated variables, coarsened copies of other columns) (`REDUNDANT_COLUMNS`).

### 4.5 Data-dependent steps (fit on the training set only, then applied to the test set)
- `select_columns`: removes near-constant columns and columns with too many `NaN`s unless the `NaN` rate differs between classes.
- `fit_caps` / `apply_caps`: 99th-percentile capping of continuous columns.
- `fit_impute` / `apply_impute`: median (continuous) / mode (categorical) imputation, computed on the relevant subgroup for module questions (e.g. only diabetics for the diabetes module). *Note: the final tree model in `run.py` keeps `NaN`s and does not need the imputation; it is used by the linear/logistic baselines.*

### 4.6 `HAREHAB1` override
`HAREHAB1` (heart-attack rehabilitation) is only asked of heart-attack survivors, so it leaks the label. It is not used as a feature; instead, `forced_positive()` forces the prediction to `+1` for respondents who answered it.

---

## 5. Model (`gbdt.py`)

Final model: our own NumPy implementation of **histogram-based gradient-boosted decision trees** (Newton boosting on the logistic loss). Nothing from scikit-learn, XGBoost or LightGBM is used.

**How it works**
- **Boosting:** every round computes the per-sample gradient `g = p - y` and Hessian `h = p(1-p)` of the logistic loss at the current scores, fits a regression tree to them, and sets each leaf to the Newton step `-G / (H + lambda_leaf)` (scaled by the learning rate). The initial score is the log-odds of the positive-class rate.
- **Binning (`fit_binner`):** each feature is binned once on the training set: exact values if it has <= 63 distinct values, quantile bins otherwise. Bin 0 is reserved for `NaN`, so missing values are handled natively by the trees.
- **Splits:** trees grow level-wise. For ordered features a split is a threshold on the bins, and `NaN` goes to whichever side gives the larger gain. For unordered categorical features (`UNORDERED_CATEGORICALS`) the bins are first sorted by `G / (H + lambda_leaf)` in the node, then a threshold on that order is used. Gradient/Hessian histograms per (node, feature, bin) are built with one `np.bincount` per level.
- **Regularization:** minimum Hessian sum per child, L2 leaf penalty, row and column subsampling per tree, maximum depth.
- **Early stopping (`fit_gbdt`):** with a validation set, training stops when the validation loss has not improved for `patience=50` rounds and the best number of trees is kept (used in CV; the final model is trained on all data with a fixed number of trees).
- `feature_importance` returns the total split gain per feature.

| Hyper-parameter | Value used in `run.py` |
|---|---|
| Number of trees | 500 (early-stopping optimum in CV: 464 trees on 72% of the data) |
| Max depth | 4 |
| Learning rate | 0.05 |
| Min. Hessian per child (`min_hessian`) | 10.0 |
| L2 leaf regularization (`lambda_leaf`) | 1.0 |
| Row / column subsampling | 0.7 / 0.7 |
| Number of bins | 64 (bin 0 = `NaN`) |
| Seed | 0 |
| Decision threshold | 0.21 (F1-optimal on out-of-fold CV predictions, identical in all 5 folds) |

---

## 6. Validation protocol

We never rely on the leaderboard score alone. All choices (hyper-parameters, number of trees, threshold, ablations) were made with **5-fold cross-validation** on the training set, and the threshold was chosen on the out-of-fold predictions. `<Add: name of the script/notebook that runs CV and ablations, and how to launch it.>`

Baselines and ablations (linear regression, ridge, logistic / regularized logistic regression with the hyper-parameters used, and the effect of each preprocessing step) are reported with their metrics in the PDF report in `report/`.

---

## 7. Report

The 2-page report (plus references) is in `report/` (`report_f.pdf`, LaTeX source alongside).

---

## 8. Reproducibility notes
- Fixed seed (`seed=0`) for all randomness in `run.py`.
