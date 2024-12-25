# Sweepystats

*Because Sweepy was taken*

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://biona001.github.io/sweepystats)
[![build Actions Status](https://github.com/biona001/sweepystats/actions/workflows/CI.yml/badge.svg)](https://github.com/biona001/sweepystats/actions)
[![codecov](https://codecov.io/gh/biona001/sweepystats/graph/badge.svg?token=UJJX0JCVKK)](https://codecov.io/gh/biona001/sweepystats)
[![Downloads](https://static.pepy.tech/badge/sweepystats)](https://pepy.tech/project/sweepystats)
[![Downloads](https://static.pepy.tech/badge/sweepystats/month)](https://pepy.tech/project/sweepystats)
![versions](https://img.shields.io/pypi/pyversions/sweepystats.svg)
![PyPI - Version](https://img.shields.io/pypi/v/sweepystats)
![GitHub Release](https://img.shields.io/github/v/release/biona001/sweepystats)

`Sweepystats` is a python package for performing the statistical [sweep operation](https://hua-zhou.github.io/teaching/biostatm280-2017spring/slides/11-sweep/sweep.html) on `numpy` matrices. 

> The sweep operator is perhaps the most versatile tool of all statistical operators that not only afford solutions to the normal equation and a gamut of additional statistics, but also allow complete insight into the nature of least squares. Once mastered, the general concepts of the sweep operator allow the whole least squares process to be visualized. 
> -- James H. Goodnight (1979)

## Installation

```shell
pip install sweepystats
```

## Documentation

Visit [https://biona001.github.io/sweepystats](https://biona001.github.io/sweepystats)

## Features

The following operations are supported **in-place** and **allocation-free**:

+ Matrix inversions
+ Computation of determinants
+ Checking of (strict) positive-definiteness
+ Linear regression (beta hat, variance of OLS estimator, residuals)

## Running tests

1. Git clone the repo
2. Install `pytest` via `pip3 install pytest` if you haven't already
3. Execute `pytest tests` in the top level directory of `sweepystats`

## Related packages

+ [SweepOperator.jl](https://github.com/joshday/SweepOperator.jl) in Julia
+ [sweep.operator](https://search.r-project.org/CRAN/refmans/fastmatrix/html/sweep.operator.html) in R

## References

+ [Biostats M280 lecture notes at UCLA](https://hua-zhou.github.io/teaching/biostatm280-2017spring/slides/11-sweep/sweep.html)
+ Section 7.4-7.6 of [Numerical Analysis for Statisticians](https://link.springer.com/book/10.1007/978-1-4419-5945-4) by Kenneth Lange (2010). Probably the best place to read about sweep operator.
+ [Blog post by SAS](https://blogs.sas.com/content/iml/2018/04/18/sweep-operator-sas.html)
+ [James Goodnight's awesome paper from 1978](https://www.jstor.org/stable/2683825)

## TODO
+ Stepwise regression
+ Conditional formulas for MVN
+ ANOVA, ANCOVA, and MANOVA
+ 2-stage and 3-stage least squares
+ Partial correlation
+ GLMs
+ Generalized (pseudo-) inverses
+ Cholesky decomposition
+ Benchmarks, e.g. timing comparison with `np.inv()` and `np.linalg.lstsq()`
+ Recursive tiling, see https://github.com/joshday/SweepOperator.jl/issues/9
+ Blog post
