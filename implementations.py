"""ML methods for Project 1 (CS-433), written with NumPy only."""

import numpy as np


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
