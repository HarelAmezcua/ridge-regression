import numpy as np

def ridge_regression(x,y,d, alpha=0.1):
    """
    Perform ridge regression on the data X, y with regularization parameter alpha.
    """
    n = len(x)
    X = np.ones((n,d+1))

    for i in range(1,d+1):
        X[:,i] = (x**i).reshape(-1)

    p = X.shape[1]

    I = np.eye(p)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

def predict(x, beta):
    """
    Predict the value of y given x and beta.
    """
    d = len(beta) - 1
    y = np.zeros_like(x)
    for i in range(d+1):
        y += beta[i] * x**i
    return y