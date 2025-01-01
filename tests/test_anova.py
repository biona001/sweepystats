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
    assert one_way.column_map == {'Intercept': [0], 'Group': [1, 2]}

    # correctness
    model = ols('Outcome ~ Group', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)  # Type III ANOVA
    f_stat, pval = one_way.f_test("Intercept")
    assert np.allclose(anova_table["F"].Intercept, f_stat)
    assert np.allclose(anova_table["PR(>F)"].Intercept, pval)
    f_stat, pval = one_way.f_test("Group")
    assert np.allclose(anova_table["F"].Group, f_stat)
    assert np.allclose(anova_table["PR(>F)"].Group, pval)
    assert np.allclose(anova_table["sum_sq"].Residual, one_way.sum_sq())

def test_kway_with_interaction():
    data = pd.DataFrame({
        'Outcome': [3.6, 3.5, 4.2, 2.7, 4.1, 5.2, 3.0, 4.8, 4.0],
        'Group': pd.Categorical(["A", "A", "B", "B", "A", "C", "B", "C", "C"]), 
        'Factor': pd.Categorical(["X", "X", "Y", "X", "Y", "Y", "X", "Y", "X"])
    })

    formula = "Outcome ~ Group + Factor + Group:Factor"
    two_way = sw.ANOVA(data, formula)
    two_way.fit()

    # data structure
    assert two_way.n == 9
    assert two_way.p == 6
    assert two_way.column_map == {'Intercept': [0], 'Group': [1, 2], 'Factor': [3], 'Group:Factor': [4, 5]}

    # correctness
    model = ols('Outcome ~ Group + Factor + Group:Factor', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)  # Type III ANOVA
    f_stat, pval = two_way.f_test("Intercept")
    assert np.allclose(anova_table["F"].Intercept, f_stat)
    assert np.allclose(anova_table["PR(>F)"].Intercept, pval)
    f_stat, pval = two_way.f_test("Group")
    assert np.allclose(anova_table["F"].Group, f_stat)
    assert np.allclose(anova_table["PR(>F)"].Group, pval)
    f_stat, pval = two_way.f_test("Factor")
    assert np.allclose(anova_table["F"].Factor, f_stat)
    assert np.allclose(anova_table["PR(>F)"].Factor, pval)
    f_stat, pval = two_way.f_test("Group:Factor")
    assert np.allclose(anova_table.loc["Group:Factor", "F"], f_stat)
    assert np.allclose(anova_table.loc["Group:Factor", "PR(>F)"], pval)
    assert np.allclose(anova_table["sum_sq"].Residual, two_way.sum_sq())
