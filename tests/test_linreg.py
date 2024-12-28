import numpy as np
import sweepystats as sw
import pytest
from scipy.stats import f

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

def test_stepwise_regression():
    # data
    n, p, k = 20, 5, 3
    X = np.random.normal(n, p, size=(n, p))
    beta = np.zeros(p)
    beta[np.random.choice(np.arange(p), size=k, replace=False)] = np.random.randn(k)
    y = X @ beta + np.random.normal()

    # fit
    ols = sw.LinearRegression(X, y)
    ols.fit()

    # f-stat and p-val for 1st variable
    f_stat, pval = ols.f_test(0)

    # least squares solution
    _, resid_full, _, _ = np.linalg.lstsq(X, y)
    _, resid_reduced, _, _ = np.linalg.lstsq(X[:, 1:], y)
    f_stat_true = (resid_reduced - resid_full) / resid_full * (n - p)
    pval_true = f.sf(f_stat_true, 1, n - p)
    assert np.allclose(f_stat, f_stat_true)
    assert np.allclose(pval, pval_true)
