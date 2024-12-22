import numpy as np
import sweepystats as sw
import pytest

def test_linreg():
    n, p = 5, 3
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    ols = sw.LinearRegression(X, y)
    ols.fit()

    # least squares solution by QR
    beta, resid, _, _ = np.linalg.lstsq(X, y)
    sigma2 = resid[0] / (n - p)
    beta_cov = sigma2 * np.linalg.inv(X.T @ X)
    beta_std = np.sqrt(np.diag(beta_cov))
    TSS = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (resid / TSS)

    assert np.allclose(ols.coef(), beta)         # beta hat
    assert np.allclose(ols.resid(), resid)       # residual
    assert np.allclose(ols.cov(), beta_cov)      # Var(beta hat)
    assert np.allclose(ols.coef_std(), beta_std) # std of beta hat
    assert np.allclose(ols.R2(), R2)             # R2

def test_high_dimensional():
    n, p = 5, 10
    X = np.random.rand(n, p)
    y = np.random.rand(n)
    ols = sw.LinearRegression(X, y)

    # X'X is singular, so there must be 1 eigenvalue that is 0
    with pytest.raises(ZeroDivisionError):
        ols.fit()