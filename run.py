"""Train L2-regularized logistic regression on the MICHD data and write submission.csv.

Steps:
    1. Load the data and map the labels from {-1, 1} to {0, 1}.
    2. Preprocess: turn special codes (7/9/77/99... and 88) into NaN or 0,
       drop mostly-missing columns, add "was missing" indicators, one-hot encode
       categorical columns, fill gaps with the median, standardize, add a bias column.
    3. Train with Newton's method. Choose lambda and the decision threshold with
       5-fold cross-validation (F1 score).
    4. Retrain on all the training data and predict the test set.
"""

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import reg_logistic_regression, reg_logistic_regression_newton

DATA_PATH = "data/dataset"
MAX_MISSING = 0.9  # drop columns with more than 90% missing values
LAMBDAS = [1e-4, 1e-3, 1e-2]
GAMMA = 0.5
MAX_ITERS = 300  # gradient descent steps
NEWTON_ITERS = 6  # Newton steps (the loss stops changing after about 6)
MAX_CATEGORIES = 20  # columns with 3 to 20 distinct values are one-hot encoded
K_FOLDS = 5
THRESHOLDS = np.arange(0.05, 0.95, 0.01)
SEED = 1

# Switches for the ablation study (all True = our final model)
USE_NEWTON = True  # train with Newton's method instead of gradient descent
MISSING_INDICATORS = True  # add a 0/1 "was missing" column per feature with gaps
ONE_HOT = True  # one-hot encode categorical columns
DROP_ADMIN = True  # drop survey bookkeeping columns (dates, ids, weights)
SQUARED_TERMS = True  # add x^2 for continuous columns

# Survey bookkeeping columns: interview date, record ids, sampling weights
ADMIN_COLUMNS = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALCOR", "_LLCPWT",
]

# BRFSS special codes, which depend on how many digits a column uses.
# We recognize the format of a column by its largest value:
#   max value -> (codes meaning "don't know" / "refused", codes meaning "none")
SPECIAL_CODES = {
    9: ([7, 9], []),
    99: ([77, 99], [88]),
    999: ([777, 999], [888]),
    9999: ([7777, 9999], []),
    99900: ([99900], []),
}


def read_column_names(data_path):
    """Feature names from the header of x_train.csv (without the Id column)."""
    with open(f"{data_path}/x_train.csv") as f:
        return np.array(f.readline().strip().split(",")[1:])


def clean_codes(x_train, x_other):
    """Replace special codes by NaN ("don't know", "refused") or 0 ("none")."""
    x_train, x_other = x_train.copy(), x_other.copy()
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        col_max = np.nanmax(col) if not np.all(np.isnan(col)) else np.nan
        if col_max not in SPECIAL_CODES:
            continue
        missing_codes, zero_codes = SPECIAL_CODES[col_max]

        # In 1-digit columns, 7 is only a code if it comes after a gap
        # (e.g. GENHLTH uses 1-5, 7, 9), otherwise it is a real category
        # (e.g. EMPLOY1 uses 1-9 and 7 means "retired")
        if col_max == 9 and np.any(col == 6):
            missing_codes = [9]

        for x in (x_train, x_other):
            x[np.isin(x[:, j], missing_codes), j] = np.nan
            x[np.isin(x[:, j], zero_codes), j] = 0
    return x_train, x_other


def preprocess(x_train, x_other):
    """Clean x_train and x_other using statistics computed on x_train only."""
    x_train, x_other = clean_codes(x_train, x_other)

    # Keep the columns that are mostly filled in
    keep = np.mean(np.isnan(x_train), axis=0) <= MAX_MISSING
    x_train, x_other = x_train[:, keep], x_other[:, keep]

    # New columns to append at the end (missing indicators, one-hot columns)
    extra_train, extra_other = [], []

    # A skipped question is information too: record where values were missing
    if MISSING_INDICATORS:
        has_nan = np.isnan(x_train).any(axis=0)
        extra_train.append(np.isnan(x_train[:, has_nan]) * 1.0)
        extra_other.append(np.isnan(x_other[:, has_nan]) * 1.0)

    # Number of distinct values: few means categorical, many means continuous
    n_values = np.array([len(np.unique(c[~np.isnan(c)])) for c in x_train.T])
    is_cat = (n_values >= 3) & (n_values <= MAX_CATEGORIES)
    is_continuous = n_values > MAX_CATEGORIES

    # Replace each categorical column by one 0/1 column per category, so that
    # e.g. marital status 5 is not treated as "more" than marital status 1
    if ONE_HOT:
        for j in np.where(is_cat)[0]:
            values = np.unique(x_train[~np.isnan(x_train[:, j]), j])
            # NaN equals no value, so a missing entry gets all zeros
            extra_train.append((x_train[:, [j]] == values) * 1.0)
            extra_other.append((x_other[:, [j]] == values) * 1.0)
        x_train, x_other = x_train[:, ~is_cat], x_other[:, ~is_cat]
        is_continuous = is_continuous[~is_cat]

    # Replace missing values by the column median
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_other = np.where(np.isnan(x_other), medians, x_other)

    # Squared terms let a linear model follow curved effects, e.g. a risk that
    # grows faster and faster with age. We square the distance to the mean.
    if SQUARED_TERMS:
        center = x_train[:, is_continuous].mean(axis=0)
        extra_train.append((x_train[:, is_continuous] - center) ** 2)
        extra_other.append((x_other[:, is_continuous] - center) ** 2)

    x_train = np.hstack([x_train] + extra_train)
    x_other = np.hstack([x_other] + extra_other)

    # Standardize, dropping constant columns (std = 0)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    varying = std > 0
    x_train = (x_train[:, varying] - mean[varying]) / std[varying]
    x_other = (x_other[:, varying] - mean[varying]) / std[varying]

    # Add a column of ones for the bias term
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_other = np.c_[np.ones(len(x_other)), x_other]
    return x_train, x_other


def train(y, tx, lambda_):
    """Fit regularized logistic regression starting from zero weights."""
    initial_w = np.zeros(tx.shape[1])
    if USE_NEWTON:
        w, _ = reg_logistic_regression_newton(y, tx, lambda_, initial_w, NEWTON_ITERS)
    else:
        w, _ = reg_logistic_regression(y, tx, lambda_, initial_w, MAX_ITERS, GAMMA)
    return w


def predict_proba(tx, w):
    """Predicted probability of class 1."""
    return 1.0 / (1.0 + np.exp(-tx.dot(w)))


def f1_score(y_true, y_pred):
    """F1 = 2 * precision * recall / (precision + recall), labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn)


def cross_validation(x, y):
    """Return the mean F1 over the K folds, shape (len(LAMBDAS), len(THRESHOLDS))."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y)), K_FOLDS)

    f1s = np.zeros((len(LAMBDAS), K_FOLDS, len(THRESHOLDS)))
    for k in range(K_FOLDS):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K_FOLDS) if j != k])

        # Preprocess once per fold; every lambda reuses the same features
        tx_train, tx_val = preprocess(x[train_idx], x[val_idx])
        for l, lambda_ in enumerate(LAMBDAS):
            w = train(y[train_idx], tx_train, lambda_)
            probs = predict_proba(tx_val, w)
            for i, t in enumerate(THRESHOLDS):
                f1s[l, k, i] = f1_score(y[val_idx], (probs >= t) * 1)
    return f1s.mean(axis=1)


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    y = (y_train + 1) / 2  # {-1, 1} -> {0, 1}

    if DROP_ADMIN:
        names = read_column_names(DATA_PATH)
        keep = ~np.isin(names, ADMIN_COLUMNS)
        x_train, x_test = x_train[:, keep], x_test[:, keep]
    print(f"Train {x_train.shape}, test {x_test.shape}, positives {y.mean():.2%}")

    # Grid search: CV F1 for every (lambda, threshold) pair, then keep the best pair
    mean_f1 = cross_validation(x_train, y)
    for l, lambda_ in enumerate(LAMBDAS):
        i = np.argmax(mean_f1[l])
        print(f"lambda={lambda_:.0e}  best threshold={THRESHOLDS[i]:.2f}  F1={mean_f1[l, i]:.4f}")
    l, i = np.unravel_index(np.argmax(mean_f1), mean_f1.shape)
    best_lambda, best_t = LAMBDAS[l], THRESHOLDS[i]
    print(f"Chosen: lambda={best_lambda:.0e}, threshold={best_t:.2f}, CV F1={mean_f1[l, i]:.4f}")

    # Retrain on the full training set and predict the test set
    tx_train, tx_test = preprocess(x_train, x_test)
    w = train(y, tx_train, best_lambda)
    y_pred = np.where(predict_proba(tx_test, w) >= best_t, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"Wrote submission.csv ({np.mean(y_pred == 1):.2%} predicted positive)")


if __name__ == "__main__":
    main()
