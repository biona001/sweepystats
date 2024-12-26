import numpy as np
import sweepystats as sw
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols

def test_oneway():
    data = pd.DataFrame({
        'Outcome': [3.6, 3.5, 4.2, 2.7, 4.1, 5.2, 3.0, 4.8, 4.0],
        'Group': pd.Categorical(["A", "A", "B", "B", "A", "C", "B", "C", "C"]), 
        'Factor': pd.Categorical(["X", "X", "Y", "X", "Y", "Y", "X", "Y", "X"])
    })

    formula = "Outcome ~ Group"
    one_way = sw.ANOVA(data, formula)
    one_way.fit()

    # data structure
    assert one_way.n == 9
    assert one_way.p == 3
    assert one_way.k == 3

    # correctness
    model = ols('Outcome ~ Group', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)  # Type II ANOVA
    assert np.allclose(anova_table["F"].Group, one_way.f_statistic())
    assert np.allclose(anova_table["PR(>F)"].Group, one_way.p_value())
    assert np.allclose(anova_table["sum_sq"].Residual, one_way.sum_sq())

