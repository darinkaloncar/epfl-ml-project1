"""ML methods for Project 1 (CS-433), written with NumPy only."""

import numpy as np
from dataclasses import dataclass
from typing import Callable


def least_squares(y, tx):
    """Least squares regression solved with the normal equations.

    Solves tx^T tx w = tx^T y.

    Args:
        y: labels, shape (N,).
        tx: feature matrix, shape (N, D).

    Returns:
        w: the optimal weights, shape (D,).
        loss: the MSE at w, 1/(2N) * ||y - tx w||^2.
    """
    rhs = tx.T @ y
    lin = tx.T @ tx
    w_hat = np.linalg.solve(lin,rhs)

    loss = 1/(2*tx.shape[0]) * np.transpose(y - tx @ w_hat) @ (y - tx @ w_hat)

    return (w_hat, loss.item())

def ridge_regression(y, tx, lambda_):
    """Ridge regression solved with the normal equations.

    We minimize the MSE plus an L2 penalty on the weights:
        1/(2N) * ||y - tx w||^2 + lambda_ * ||w||^2

    Setting the gradient to zero gives the linear system from the lecture:
        (tx^T tx + 2N * lambda_ * I) w = tx^T y

    Args:
        y: labels, shape (N,).
        tx: feature matrix, shape (N, D).
        lambda_: regularization strength.

    Returns:
        w: the optimal weights, shape (D,).
        loss: the MSE at w, without the penalty term.
    """
    N, D = tx.shape

    # The lecture writes the penalty in the system as lambda' = 2N * lambda
    lambda_prime = 2 * N * lambda_
    a = tx.T.dot(tx) + lambda_prime * np.eye(D)
    b = tx.T.dot(y)

    # Solving the system is cheaper and more stable than inverting a
    w = np.linalg.solve(a, b)

    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * N)
    return w, loss


def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    proj = tx @ w
    loss = np.logaddexp(0,proj) - y * proj
    return float(np.mean(loss))

def logistic_grad(y, tx, w):
    sig = sigmoid(tx @ w)
    return tx.T @ (sig - y) / tx.shape[0]

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression trained with gradient descent, for labels y in {0, 1}.

    The loss is the average negative log-likelihood:
        L(w) = 1/N * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]
    and its gradient is 1/N * tx^T (sigmoid(tx w) - y).

    Args:
        y: labels in {0, 1}, shape (N,).
        tx: feature matrix, shape (N, D).
        initial_w: starting weights, shape (D,).
        max_iters: number of gradient descent steps.
        gamma: step size.

    Returns:
        w: the weights after the last step, shape (D,).
        loss: the logistic loss at that w.
    """
    N = len(y)
    w = initial_w

    for _ in range(max_iters):
        # Predicted probability that each sample belongs to class 1
        probs = 1.0 / (1.0 + np.exp(-tx.dot(w)))
        grad = tx.T.dot(probs - y) / N
        w = w - gamma * grad

    # logaddexp(0, z) is log(1 + exp(z)), but it doesn't overflow when z is large
    z = tx.dot(w)
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression trained with gradient descent, y in {0, 1}.

    We minimize the logistic loss plus an L2 penalty on the weights:
        L(w) + lambda_ * ||w||^2
    so the gradient is the logistic gradient plus 2 * lambda_ * w.

    Args:
        y: labels in {0, 1}, shape (N,).
        tx: feature matrix, shape (N, D).
        lambda_: regularization strength.
        initial_w: starting weights, shape (D,).
        max_iters: number of gradient descent steps.
        gamma: step size.

    Returns:
        w: the weights after the last step, shape (D,).
        loss: the logistic loss at that w, without the penalty term.
    """
    N = len(y)
    w = initial_w

    for _ in range(max_iters):
        probs = 1.0 / (1.0 + np.exp(-tx.dot(w)))
        grad = tx.T.dot(probs - y) / N + 2 * lambda_ * w
        w = w - gamma * grad

    z = tx.dot(w)
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss


def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters):
    """L2-regularized logistic regression trained with Newton's method, y in {0, 1}.

    Same objective as reg_logistic_regression, but each step uses the Hessian:
        H = 1/N * tx^T S tx + 2 * lambda_ * I,  with S = diag(p * (1 - p))
        w <- w - H^{-1} * gradient
    Newton converges in a few steps (about 10) where gradient descent needs
    thousands, and the D x D Hessian is cheap when D is a few hundred.

    Args:
        y: labels in {0, 1}, shape (N,).
        tx: feature matrix, shape (N, D).
        lambda_: regularization strength.
        initial_w: starting weights, shape (D,).
        max_iters: number of Newton steps.

    Returns:
        w: the weights after the last step, shape (D,).
        loss: the logistic loss at that w, without the penalty term.
    """
    N, D = tx.shape
    w = initial_w

    for _ in range(max_iters):
        probs = 1.0 / (1.0 + np.exp(-tx.dot(w)))
        grad = tx.T.dot(probs - y) / N + 2 * lambda_ * w

        # tx.T * s multiplies column n of tx.T by s_n, i.e. tx^T S without building S
        s = probs * (1 - probs)
        hess = (tx.T * s).dot(tx) / N + 2 * lambda_ * np.eye(D)
        w = w - np.linalg.solve(hess, grad)

    z = tx.dot(w)
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss


@dataclass
class CompositeFunction:
    f: Callable
    grad: Callable
    g: Callable
    prox: Callable
    lips_grad: float

    def __call__(self, w):
        return self.f(w) + self.g(w)

@dataclass
class FISTA_state:
    w_k: np.ndarray
    y_k: np.ndarray
    t_k: float
    alpha: float

def FISTA_initialize(F, w_zero):
    return FISTA_state(w_k=w_zero, y_k=w_zero, t_k=1.0, alpha=1.0 / F.lips_grad)

def FISTA_update(F, state):
    w_k, y_k, t_k, alpha = state.w_k, state.y_k, state.t_k, state.alpha

    w_next = F.prox(alpha, y_k - alpha * F.grad(y_k))
    if (w_next - w_k) @ (y_k - w_next) > 0:  # gradient restart
        y_k, t_k = w_k, 1.0
        w_next = F.prox(alpha, y_k - alpha * F.grad(y_k))
    t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
    y_next = w_next + (t_k - 1) / t_next * (w_next - w_k)

    return FISTA_state(w_k=w_next, y_k=y_next, t_k=t_next, alpha=alpha)

@dataclass
class OptAlgorithm:
    name: str
    init_state: Callable
    state_update: Callable

FISTA = OptAlgorithm(name="FISTA", init_state=FISTA_initialize, state_update=FISTA_update)

def l1_prox(gamma, z, lambda_, penalized):
    w = z.copy()
    w[penalized] = np.sign(z[penalized]) * np.maximum(np.abs(z[penalized]) - gamma * lambda_, 0)
    return w

def l1_logistic_regression(y, tx, lambda_, initial_w, max_iters, unpenalized_idx=None):
    """FISTA on mean logistic loss + lambda_ * ||w||_1, with y in {0, 1}.
    Indices in unpenalized_idx (e.g. the bias column) are excluded from the l1 term."""
    penalized = np.ones(tx.shape[1], dtype=bool)
    if unpenalized_idx is not None:
        penalized[unpenalized_idx] = False

    F = CompositeFunction(
        f=lambda w: calculate_loss(y, tx, w),
        grad=lambda w: logistic_grad(y, tx, w),
        g=lambda w: lambda_ * np.sum(np.abs(w[penalized])),
        prox=lambda gamma, z: l1_prox(gamma, z, lambda_, penalized),
        lips_grad=np.linalg.eigvalsh(tx.T @ tx)[-1] / (4 * tx.shape[0]),
    )

    state = FISTA.init_state(F, initial_w)
    for _ in range(max_iters):
        state = FISTA.state_update(F, state)
    return (state.w_k, calculate_loss(y, tx, state.w_k))


UNORDERED_CATEGORICALS = ["MARITAL", "EMPLOY1", "RENTHOM1", "IMFVPLAC", "WHRTST10", "_PRACE1", "_MRACE1", "_RACE"]

def fit_one_hot(x, features, columns=UNORDERED_CATEGORICALS):
    """Categories of each column in `columns`, fitted on the training set."""
    return {c: np.unique(x[:, features.index(c)]) for c in columns}

def apply_one_hot(x, features, categories):
    """Replace each categorical column by one indicator column per training category.
    Categories unseen in training get all zeros."""
    keep = [j for j, name in enumerate(features) if name not in categories]
    blocks, names = [x[:, keep]], [features[j] for j in keep]
    for c, values in categories.items():
        blocks.append((x[:, [features.index(c)]] == values[None, :]).astype(float))
        names += [f"{c}={v:g}" for v in values]
    return np.hstack(blocks), names
