import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def polynomial_ridge(x,y,d, alpha=0.1):
    """ 
    Perform polynomial ridge regression of degree d on the data (x,y) with regularization parameter alpha.
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

def plot_regression(x, y, y_final):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, color='blue', s=100, label='Data points')
    sns.lineplot(x=x, y=y_final, color='red', linewidth=2.5, label='Ridge regression')
    plt.title('Ridge Regression', fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.legend()
    plt.show()