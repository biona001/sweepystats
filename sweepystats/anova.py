import sweepystats as sw
import numpy as np
import pandas as pd
import patsy
from scipy.stats import f
from .util import designate_X_columns

class ANOVA:
    """
    A class to perform (k-way) ANOVA based on the sweep operation. 

    Parameters:
    + `df`: A `pandas` dataframe containing the covariates and outcome. 
    + `formula`: A formula string to define the model, e.g. 
        'y ~ Group + Factor + Group:Factor'.
    """
    def __init__(self, df, formula):
        self.df = df
        self.formula = formula

        # Use patsy to parse the formula and build the design matrix
        y, X = patsy.dmatrices(formula, df, return_type="dataframe")
        self.X = np.array(X, order='F', dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).ravel()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

        # number of groups for each variable in RHS of formula
        self.column_map = designate_X_columns(X, formula)

        # initialize least squares instance
        self.ols = sw.LinearRegression(self.X, self.y)

    def fit(self, verbose=True):
        """Fit ANOVA model by sweep operation"""
        return self.ols.fit(verbose = verbose)

    def f_statistic(self):
        """Computes the F-statistic associated with the ANOVA model."""
        n, k = self.n, self.k # number of samples and groups
        yhat = np.matmul(self.X, self.ols.coef()) # predicted y
        ss_between = np.sum((yhat - np.mean(self.y)) ** 2) # between group sum of squares
        ss_within = self.sum_sq()
        return (ss_between / (k - 1)) / (ss_within / (n - k))

    def sum_sq(self):
        """Compuptes within-group sum of squares error"""
        return self.ols.resid()

    def p_value(self):
        n, k = self.n, self.k # number of samples and groups
        df1 = k - 1
        df2 = n - k
        return f.sf(self.f_statistic(), df1, df2)
