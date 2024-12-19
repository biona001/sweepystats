import numpy as np
import sweepy as sw

def test_linreg():
    X = np.random.rand(5, 3)
    y = np.random.rand(5)
    ols = sw.LinearRegression(X, y)
    ols.fit()
    beta, resid, _, _ = np.linalg.lstsq(X, y) # least squares solution by QR

    # beta hat
    np.allclose(ols.coef(), beta)

    # residual
    np.allclose(ols.resid(), resid)
