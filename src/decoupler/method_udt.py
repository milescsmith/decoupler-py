"""
Method UDT.
Code to run the Univariate Decision Tree (UDT) method.
"""

import numpy as np
import pandas as pd
import sklearn as sk
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from decoupler.pre import (
    extract,
    filt_min_n,
    get_net_mat,
    match,
    rename_net,
    return_data,
)


def fit_dt(sk, regulator, sample, min_leaf=5, seed=42):
    # Fit DT
    x, y = regulator.reshape(-1, 1), sample.reshape(-1, 1)
    regr = sk.tree.DecisionTreeRegressor(min_samples_leaf=min_leaf, random_state=seed)
    regr.fit(x, y, check_input=False)

    # Get importance
    return regr.tree_.compute_feature_importances(normalize=False)[0]


def udt(mat, net, min_leaf=5, seed=42, verbose=False):
    # Init empty acts
    acts = np.zeros((mat.shape[0], net.shape[1]))

    # For each sample and regulator fit dt
    for i in tqdm(range(mat.shape[0]), disable=not verbose):
        if isinstance(mat, csr_matrix):
            sample = mat[i].toarray()[0]
        else:
            sample = mat[i]
        for j in range(net.shape[1]):
            acts[i, j] = fit_dt(sk, net[:, j], sample, min_leaf=min_leaf, seed=seed)

    return acts


def run_udt(
    mat,
    net,
    source="source",
    target="target",
    weight="weight",
    min_leaf=5,
    min_n=5,
    seed=42,
    verbose=False,
    use_raw=True,
):
    """
    Univariate Decision Tree (UDT).

    UDT fits a single regression decision tree for each sample and regulator, where the observed molecular readouts in `mat`
    are the response variable and the regulator weights in `net` are the explanatory one. Target features with no associated
    weight are set to zero. The obtained feature importance from the fitted model is the activity (`udt_estimate`) of a given
    regulator.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    min_leaf : int
        The minimum number of samples required to be at a leaf node.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        UDT scores. Stored in `.obsm['udt_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['udt_pvals']` if `mat` is AnnData.
    """

    # Extract sparse matrix and array of genes
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(c, net, min_n=min_n)
    sources, targets, net = get_net_mat(net)

    # Match arrays
    net = match(c, targets, net)

    if verbose:
        pass

    # Run UDT
    estimate = udt(m, net, min_leaf=min_leaf, seed=seed, verbose=verbose)

    # Transform to df
    estimate = pd.DataFrame(estimate, index=r, columns=sources)
    estimate.name = "udt_estimate"

    return return_data(mat=mat, results=(estimate,))
