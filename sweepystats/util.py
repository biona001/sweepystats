import patsy

def designate_X_columns(X, formula):
    """
    `patsy.dmatrices` creates the (one-hot encoded) `X` from a dataframe 
    `df` and a `formula`. This function figures out which column(s) of `X`
    correspond to the right-hand side variables in `formula`. 
    """
    # get number of columns for each original covariate in X
    mapping = {}
    model = patsy.ModelDesc.from_formula(formula)
    for term in model.rhs_termlist:
        mapping[term.name()] = X.design_info.term_codings[term][0].num_columns

    # to correspond the columns, compute the cumsum
    offset = 1 if "Intercept" in mapping.keys() else 0
    ks = {}
    for k, v in mapping.items():
        if k == "Intercept":
            continue
        ks[k] = list(range(offset, offset + v))
        offset += v

    # take care of intercept term
    if "Intercept" in mapping.keys():
        term = model.rhs_termlist[1].name() # first non-intercept term
        ks[term].insert(0, 0)

    return ks