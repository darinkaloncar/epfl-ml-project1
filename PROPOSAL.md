# Project 1 proposal: predicting MICHD from the BRFSS data

The lecture tags below are the PDF names: L01b regression, L01c loss functions, L02a optimization, L03a least squares, L03b over/underfitting, L03c maximum likelihood, L03d ridge and lasso, L04a model selection and CV, L04b bias–variance, L05a classification, L05b logistic regression, L06a SVM, L06b k-NN and the curse of dimensionality.

**Not covered in L01–L06:** F1, precision and recall, imputing missing values, one-hot encoding, reweighting classes and stratified splits. Where I use one of these, I mark it **[not in lectures]** so you can cite it separately in the report.

## 0. Facts I checked in the data

| | |
|---|---|
| Train / test rows | 328,135 / 109,379 |
| Features | 321 (plus `Id`) |
| Positives (`_MICHD = 1`) | 28,975, or **8.83%** |
| Missing values per column | 82 columns complete · 57 under 10% · 35 at 10–50% · 48 at 50–90% · **99 at 90% or more** |
| Special codes | `7/77/777` means "don't know", `9/99/9999` means "refused", `88` means "none". For example, `PHYSHLTH=88` (204k rows) means 0 days, and `WEIGHT2=9999` (13k rows) means refused. |
| Leakage | The columns used to define MICHD (`CVDINFR4`, `CVDCRHD4`) are not in the data. Strong related features such as `CVDSTRK3`, `_RFHYPE5`, `_RFCHOL` and `_AGEG5YR` are present. |

## 1. Understanding and preprocessing the data

**What to check**
- **Missing values.** Compute the fraction missing in each column. Many columns are missing because the survey skipped the question for that person, so the gap itself carries information.
- **Coded values.** Any value like `9999` is a code, not a measurement. Fed in as a number, it becomes an extreme outlier. Squared loss is very sensitive to outliers, so these would distort least squares (L01c "Outliers").
- **Categorical vs. continuous.** Use the number of distinct values to tell them apart. For example, `GENHLTH` takes values 1–5 and `_STATE` is a nominal code, while `_BMI5` and `PHYSHLTH` are continuous.
- **Duplicated features.** BRFSS has both raw columns (`WEIGHT2`) and cleaned versions (`_BMI5`, `_AGEG5YR`, `_RF*`). Nearly identical columns make XᵀX ill-conditioned (L03a "Rank deficiency and ill-conditioning").
- **Class imbalance.** At 8.8% positives, the plain least-squares classifier's decision line depends heavily on how many points each class has (L05a "Sensitivity to unbalanced data").
- **Low-value columns.** Constant columns and administrative ones (`IDATE`, `SEQNO`, `FMONTH`, `_PSU`) add no signal, only variance (L04b).

**What to do with it**
1. Drop identifier, date and administrative columns.
2. Map the special codes: "don't know" and "refused" become missing, and `88` becomes 0.
3. Drop columns missing more than τ of their values. Treat τ as a hyperparameter (for example 0.5, 0.8 or 0.9) chosen by cross-validation (L04a "Model selection").
4. Fill the remaining gaps with the **training-fold** median (continuous) or most frequent value (categorical), and add a 0/1 "was missing" indicator **[not in lectures]**.
5. For categorical columns with few levels, use one-hot encoding **[not in lectures]**. Otherwise a linear model treats state codes as ordered numbers. This is a form of expanding the feature vector (L03b "Extended / augmented feature vectors").
6. Clip or log-transform heavy-tailed continuous features to limit outliers (L01c). Huber loss is the other option from L01c, but clipping is simpler.
7. **Standardize** every feature to zero mean and unit variance, using statistics from the training fold only. Gradient descent is very sensitive to badly scaled features (L02a "Feature normalization and pre-conditioning"), and the ridge penalty only treats all weights fairly if features are on the same scale (L03d).
8. Add a column of ones for the bias term (L01b, L05b "What about the bias term?").
9. **Feature expansion.** Add polynomial terms of degree d, only for the continuous features (L03b, L04a). Optionally add a few hand-picked products, such as age × hypertension. Full degree-2 expansion of about 300 features gives about 45k columns, which is too expensive.

## 2. Candidate models

| Model | Lecture | Pros on this dataset | Cons on this dataset |
|---|---|---|---|
| **Least squares** on ±1 labels | L03a, L05a | Closed form and fast (`np.linalg.solve`), a good baseline | Outputs are not probabilities; sensitive to imbalance and extreme values (L05a A–C); unstable when XᵀX is ill-conditioned |
| **MSE with GD / SGD** | L02a | The required functions; SGD scales to 328k rows | Same loss as above, plus step size to tune; mainly useful to check against the closed form |
| **Ridge regression** | L03d | λI lifts the eigenvalues, which fixes the collinear one-hot and duplicate columns; MAP view (L03d) | Still squared loss on ±1 labels, so still sensitive to imbalance |
| **Logistic regression** | L05b | Outputs P(y=1\|x) directly, which is what threshold tuning needs (§4); convex; more robust to imbalance and extreme values than LS (L05b comparison plots) | No closed form; GD needs a good γ; weights can diverge if the data is separable, which becomes possible after a large feature expansion |
| **Regularized logistic regression** (L2) | L05b, L03d | Bounded weights, stable optimization, and less overfitting with many one-hot or polynomial features | λ needs tuning. **Expected best model.** |
| **Lasso / L1 penalty** | L03d | Sparse weights pick out features, which helps interpretation in the report | Not differentiable, so it needs subgradients (L02a); optional |
| **Soft-margin SVM** (hinge + L2) | L06a | Hinge is another convex surrogate for 0-1 loss (L05a); stochastic subgradient training is cheap | No probabilities, so the threshold must be set on the raw score; dual has an N×N kernel matrix, impossible at N = 328k |
| **k-NN** | L05a, L06b | No training; flexible decision boundary | Slow at query time (100k test × 328k train); high dimension hurts it (L06b "Curse of dimensionality"); poor fit here. Use only as a discussion point or on a small subsample |

Two more points from L05b and L02a:
- **Newton's method** for logistic regression is affordable at D ≈ a few hundred, since the Hessian is XᵀSX/N, D×D. It converges much faster than GD and is a good extra optimizer to compare.
- The required `logistic_regression` uses y ∈ {0,1}. Map the labels with y₀₁ = (y+1)/2, then map predictions back to {−1, 1} before submitting.

## 3. Training

**Optimizer**
- Least squares and ridge: normal equations via `np.linalg.solve`, never an explicit inverse (L03a additional notes).
- Logistic models: full-batch GD. The gradient Xᵀ(σ(Xw) − y)/N costs O(ND), which is fine at this size (L05b, L02a "Computational cost").
- Mini-batch SGD (L02a) only if you use large feature expansions. Optionally compare with damped Newton (L05b).

**Hyperparameters.** Choose them by grid search (L02a, L04a) scored by K-fold CV (§4):

| Hyperparameter | Grid | How to pick it |
|---|---|---|
| γ (step size) | 0.01 to 1, log scale | Too large diverges, too small is slow (L02a "Step-size and stopping"). Standardized features let one γ work for all coordinates. Pick the largest γ whose training loss decreases steadily. |
| max_iters | — | Plot training and validation loss against iterations (L04b "Draw learning curve!"). Stop where validation loss levels off. Stopping early also regularizes (L03d additional notes). |
| λ | 10⁻⁶ to 10⁰, log scale | Plot the validation-score curve against λ (L04a/L04b "Model selection curves"). |
| d (polynomial degree) | 1, 2, 3 | Tune jointly with λ. Expect a U-shaped bias–variance curve (L04b). |
| τ (missingness cut-off) | 0.5, 0.8, 0.9 | A preprocessing hyperparameter, scored the same way. |

L04a also shows that testing K candidate settings only adds a √ln K term to the error bound. A moderate grid is therefore cheap in statistical terms.

**Class imbalance.** These options run from most to least grounded in the lectures:
1. **Tune the decision threshold** (§4). In logistic regression, changing the bias w₀ moves the decision boundary (L05b "Changing w₀ shifts the decision region"). Moving the threshold away from 0.5 is equivalent to changing w₀. This is the simplest fix and should come first.
2. **Weight the loss by class.** Multiply each positive example's loss by about N₋/N₊ ≈ 10 **[not in lectures]**. It fits directly into the sum-over-examples form of the loss (L02a "Sum objectives").
3. **Undersample the negatives** when training **[not in lectures]**. Validate on the original class distribution.

## 4. Evaluation

**Validation setup**
- Hold out about 20% once as a final check. Never use it for tuning.
- Run **5-fold CV** on the remaining 80% (L04a "M-fold Cross-Validation"). It gives a mean score and its spread across folds.
- Fit every preprocessing statistic (medians, means and standard deviations, which columns to drop) **inside each training fold** only. Otherwise information from the validation fold leaks into training.
- Keep each fold at about 8.8% positives **[stratified splits: not in lectures]**.

**How precise is the validation score?** Hoeffding's bound from L04a, with a validation fold of about 65k rows and δ = 0.05:
- Accuracy is within about ±0.5% of the true value.
- With K = 100 hyperparameter settings tried, it is within about ±0.8%.

Smaller differences between methods are noise. Report the CV standard deviation next to every number. F1 is not an average of per-example losses, so Hoeffding does not apply to it directly. Use the spread across folds instead.

**Why accuracy alone misleads.** The classifier that always predicts −1 scores **91.2% accuracy** and is useless. It has zero recall and an F1 of 0. Accuracy is the 0-1 risk (L05a), and with 91% negatives it is driven almost entirely by the negatives.

F1 = 2PR/(P+R), where P is precision and R is recall **[not in lectures]**. F1 ignores true negatives, so it rewards finding the positives.

Report F1 as the main metric and also show accuracy, precision and recall.

**Choosing the decision threshold**
- The Bayes classifier predicts argmax P(y|x), which is a threshold of 0.5 (L05a "Bayes classifier"). That rule minimizes **0-1 loss**, not F1.
- With 8.8% positives, few people reach p ≥ 0.5, so recall at that threshold will be very low.
- Instead, scan the threshold t from 0.05 to 0.95 on each validation fold's predicted probabilities. Pick the t that maximizes mean F1 across folds, then retrain on all the training data and apply that t to the test set.
- For least squares or SVM, scan the raw score the same way.
- Expect the best t to land roughly between 0.15 and 0.3.

## 5. Ablation plan for the report

**Baselines**
1. Always predict −1: accuracy about 91%, F1 = 0. This shows why F1 is needed.
2. Least squares on minimally cleaned features, with threshold 0.
3. Ridge regression with tuned λ.
4. Plain logistic regression with threshold 0.5.

**Changes to measure.** Add one at a time on top of the best model, and report Δ in F1 (± CV std) for each:

| # | Change | Lecture |
|---|---|---|
| A | Special codes → missing, `88` → 0 | L01c |
| B | Drop columns above τ missing, plus median imputation and missing indicators | [not in lectures] / L04a |
| C | Standardization (measure the effect on GD convergence speed too) | L02a |
| D | One-hot encoding of categorical features | L03b |
| E | Squared loss → logistic loss | L05a, L05b |
| F | Add the L2 penalty, with λ tuned | L03d, L05b |
| G | Tune the threshold instead of using 0.5 | L05a, L05b |
| H | Class weighting | [not in lectures] |
| I | Polynomial degree 2 on continuous features | L03b, L04b |
| J | (Optional) GD vs. Newton, or the hinge loss | L05b, L06a |

Present it as a table of cumulative improvements with ± std, plus a λ/d validation curve and a threshold-vs-F1 curve. Also do a "leave one out" pass: remove each change from the final pipeline in turn. The change whose removal hurts F1 most is your headline finding, which the grading criteria ask for.

## Recommended order of experiments

1. **Infrastructure.** Implement the 6 required functions and pass `grading_tests`. Write the K-fold CV helper and the F1, accuracy, precision and recall functions.
2. **Trivial baseline.** Always predict −1, to get the accuracy vs. F1 numbers.
3. **Minimal pipeline.** Drop high-missing columns, median-impute, standardize, add the bias column. Fit least squares and ridge, and tune the threshold on CV.
4. **Logistic regression.** Same features, GD with γ chosen from learning curves, first at threshold 0.5 and then with the tuned threshold. Expect a large F1 jump; make your first leaderboard submission here.
5. **Regularized logistic regression.** Grid over λ.
6. **Cleaning.** Special codes, missing indicators and one-hot encoding. Redo the λ grid.
7. **Class weighting.** Compare it against threshold tuning, and against both together.
8. **Feature expansion.** Degree 2 or 3 on continuous features, tuning d jointly with λ and watching for the U-shaped curve.
9. **Optional extras.** Newton's method, L1 penalty for feature selection, SVM.
10. **Final checks.** Run once on the untouched 20% hold-out, then the leave-one-out ablation. Freeze `run.py` with a fixed random seed for reproducibility.
