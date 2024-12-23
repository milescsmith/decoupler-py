"""
Utility functions to benchmark methods and nets.
Functions to benchmark methods and nets using perturbation experiments.
"""

import numpy as np
import pandas as pd
import scipy
from numpy.random import default_rng
from scipy.sparse import issparse

from decoupler.metrics import (
    metric_auprc,
    metric_auroc,
    metric_mcauprc,
    metric_mcauroc,
    metric_nrank,
    metric_rank,
    metric_recall,
)
from decoupler.pre import match
from decoupler.utils import get_toy_data


def get_toy_benchmark_data(n_samples=24, seed=42, shuffle_perc=0.25):
    """
    Generate a toy mat, net and obs for testing the benchmark pipeline.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed to use.
    shuffle_perc : float
        Percentage of the ground truth to randomize.

    Returns
    -------
    mat : DataFrame
        mat example.
    net : DataFrame
        net example.
    obs : DataFrame
        obs example.
    """

    # Get toy data
    mat, net = get_toy_data(n_samples=n_samples, seed=seed)

    # Simulate 2 populations of perturbations
    obs = pd.DataFrame(columns=["group"])
    n_samples = mat.shape[0]
    n = int(n_samples / 2)
    res = n_samples % 2
    obs["perturb"] = [["T1", "T2"] for _ in range(n)] + [
        ["T3", "T4"] for _ in range(n + res)
    ]
    obs["group"] = np.tile(["CA", "CB"], n + res)[: obs["perturb"].size]
    obs["sign"] = 1
    obs.index = mat.index.copy()

    # Shuffle a percentage of the samples
    idxs = np.arange(mat.shape[0])
    rng = default_rng(seed=seed)
    idxs = rng.choice(idxs, int(idxs.size * shuffle_perc), replace=False)
    r_idxs = rng.choice(idxs, idxs.size, replace=False)
    mat.iloc[r_idxs] = mat.iloc[idxs].values

    return mat, net, obs


def show_metrics():
    """
    Shows available evaluation metrics.
    The first column correspond to the function name in decoupler and the
    second to the metrics's full name.

    Returns
    -------
    df : DataFrame
        Dataframe with the available metrics.
    """

    import decoupler

    lst = dir(decoupler)
    return pd.DataFrame(
        [
            {
                "Function": m,
                "Name": getattr(decoupler, m).__doc__.split("\n")[1].lstrip(),
            }
            for m in lst
            if m.startswith("metric_")
        ]
    )


def validate_metrics(metrics):
    # Check if not list
    if type(metrics) is str:
        metrics = [metrics]

    # Retrieve available metrics
    a_metrics = [metric.split("metric_")[1] for metric in show_metrics()["Function"]]

    # Check if given metrics exist
    for metric in metrics:
        if metric not in a_metrics:
            msg = (
                f"Metric {metric} not available, please run show_metrics() "
                f"to see the list of available metrics."
            )
            raise ValueError(msg)


def compute_metric(act, grt, metric, pi0=0.5, n_iter=1000, seed=42):
    if metric == "rank":
        scores = metric_rank(grt, act)
        ci = np.nan
    elif metric == "nrank":
        scores = metric_nrank(grt, act)
        ci = np.nan
    else:
        # Flatten across obs
        act = act.ravel()
        grt = grt.ravel()
        # Identify activity scores with NAs
        nan_mask = np.isnan(act)
        # Remove NAs from activity matrix and ground truth
        act = act[~nan_mask]
        grt = grt[~nan_mask]
        # Compute Class Imbalance
        ci = np.sum(grt) / len(grt)
        if metric == "auroc":
            scores = metric_auroc(grt, act)
        elif metric == "auprc":
            scores = metric_auprc(grt, act, pi0=pi0)
        elif metric == "mcauroc":
            scores = metric_mcauroc(grt, act, n_iter=n_iter, seed=seed)
        elif metric == "mcauprc":
            scores = metric_mcauprc(grt, act, n_iter=n_iter, seed=seed)
        elif metric == "recall":
            scores = metric_recall(grt, act)

    # Output must be list
    if not isinstance(scores, np.ndarray):
        scores = np.array([scores])

    return scores, ci


def append_by_experiment(
    df,
    grpby_i,
    grp,
    act,
    grt,
    mthds,
    metrics,
    pi0=0.5,
    n_iter=1000,
    seed=42,
):
    # Compute per method and metric
    for m in range(len(mthds)):
        mth = mthds[m]
        act_i = act[:, :, m]
        # Compute metrics
        for metric in metrics:
            scores, ci = compute_metric(
                act_i, grt, metric, pi0=pi0, n_iter=n_iter, seed=seed
            )
            for score in scores:
                row = [grpby_i, grp, None, mth, metric, score, ci]
                df.append(row)


def append_by_source(
    df,
    grpby_i,
    grp,
    act,
    grt,
    srcs,
    mthds,
    metrics,
    min_exp=5,
    pi0=0.5,
    n_iter=1000,
    seed=42,
):
    for m in range(len(mthds)):
        # Extract per method
        mth = mthds[m]
        act_i = act[:, :, m]
        # Remove sources with less than min_exp
        src_msk = np.sum(grt > 0.0, axis=0) >= min_exp
        act_i, grt_i = act[:, src_msk, :], grt[:, src_msk]
        srcs_method = srcs[src_msk]
        # Compute per source, method and metric
        for s in range(len(srcs_method)):
            src = srcs_method[s]
            grt_source = grt_i[:, s]
            act_source = act_i[:, s, m]
            # Check that grt is not all the same
            unq_grt = np.unique(grt_source[~np.isnan(act_source)])
            # Convert from vector to arr
            grt_source, act_source = grt_source[np.newaxis], act_source[np.newaxis]
            if unq_grt.size > 1:
                for metric in metrics:
                    scores, ci = compute_metric(
                        act_source,
                        grt_source,
                        metric,
                        pi0=pi0,
                        n_iter=n_iter,
                        seed=seed,
                    )
                    for score in scores:
                        row = [grpby_i, grp, src, mth, metric, score, ci]
                        df.append(row)


def append_metrics_scores(
    df,
    grpby_i,
    grp,
    act,
    grt,
    srcs,
    mthds,
    metrics,
    by,
    min_exp=5,
    pi0=0.5,
    n_iter=1000,
    seed=42,
):
    if not min_exp > 0:
        msg = "Argument min_exp must be bigger than 0."
        raise ValueError(msg)

    if by == "experiment":
        append_by_experiment(
            df,
            grpby_i,
            grp,
            act,
            grt,
            mthds,
            metrics,
            pi0=pi0,
            n_iter=n_iter,
            seed=seed,
        )

    elif by == "source":
        append_by_source(
            df,
            grpby_i,
            grp,
            act,
            grt,
            srcs,
            mthds,
            metrics,
            min_exp=min_exp,
            pi0=pi0,
            n_iter=n_iter,
            seed=seed,
        )


def adjust_sign(mat, v_sign):
    v_sign = v_sign.reshape(-1, 1)
    if issparse(mat):
        mat = mat.multiply(v_sign).tocsr()
    else:
        mat = mat * v_sign
    return mat


def filter_act_by_pval(m, res, use_pval):
    if use_pval is not None:
        estimate_name = m.split("_")[0]
        act = res[m].values
        pval_name = estimate_name + "_pvals"
        if pval_name in res:
            pval = res[pval_name].values
            pval = scipy.stats.false_discovery_control(pval, axis=1)
            act[(pval >= use_pval) | (act < 0)] = 0.0
        else:
            q = np.quantile(act, 1 - use_pval, axis=1).reshape(-1, 1)
            q = np.clip(q, a_min=0, a_max=None)  # Remove negative acts
            act[act < q] = 0.0
        res[m].loc[:, :] = act


def build_acts_tensor(res, use_pval):
    # Get unique methods
    mthds = [m for m in res.keys() if "_pvals" not in m]

    # Extract dimensions
    exps = res[mthds[0]].index.values
    srcs = res[mthds[0]].columns.values

    # Build acts tensor and sort by exps and srcs
    n_exp, n_src, n_mth = len(exps), len(srcs), len(mthds)
    acts = np.zeros((n_exp, n_src, n_mth))
    for i, m in enumerate(mthds):
        filter_act_by_pval(m, res, use_pval)
        acts[:, :, i] = res[m].values
    msk = np.argsort(srcs)
    acts = acts[:, msk]
    srcs = srcs[msk]
    msk = np.argsort(exps)
    exps = exps[msk]

    return acts, exps, srcs, mthds


def build_grts_mat(obs, exps, srcs):
    # Explode nested perturbs and pivot into mat
    grts = obs.explode("perturb").pivot(columns="perturb", values="sign").fillna(0.0)

    # Sort by columns (srcs) and by rows (exps)
    msk = np.argsort(grts.columns)
    grts = grts.loc[exps].iloc[:, msk]

    # Remove cols that are not in res srcs
    msk = np.isin(grts.columns.values, srcs)
    grts = grts.loc[:, msk]

    return grts


def unique_obs(col):
    # Gets unique categories from a column with both lists and elements.

    # Init empty cats
    cats = set()

    for row in col:
        # Check if col elements are lists
        if type(row) is list:
            for r in row:
                if r not in cats:
                    cats.add(r)
        elif row not in cats:
            cats.add(row)

    return np.sort(list(cats))


def build_msks_tensor(obs, groupby):
    # If groupby
    if groupby is not None:
        # Init empty lsts
        msks = []
        grps = []
        grpbys = []
        for _ in groupby:
            # Handle nested groupbys
            if isinstance(_, list):
                grpby_i = np.sort(_)
                grpby_name = "|".join(np.sort(grpby_i))
                if grpby_i.size > 1:
                    obs[grpby_name] = obs[grpby_i[0]].str.cat(obs[grpby_i[1:]], sep="|")
                grpby_i = grpby_name
            else:
                grpby_i = _

            # Find msk in obs based on groupby
            grps_j = unique_obs(obs[grpby_i].values)
            msk_i = []
            grps_i = []
            for grp in grps_j:
                m = np.array([grp in lst for lst in obs[grpby_i]])
                msk_i.append(m)
                grps_i.append(grp)

            # Append
            msks.append(msk_i)
            grpbys.append(grpby_i)
            grps.append(grps_i)

    else:
        msks = None
        grpbys = None
        grps = None

    return msks, grpbys, grps


def format_acts_grts(res, obs, groupby, use_pval):
    # Build acts tensor and sort by exps and srcs
    acts, exps, srcs, mthds = build_acts_tensor(res, use_pval)

    # Make sure obs and acts match by exps idxs
    obs = obs.loc[exps]

    # Build sorted and filtered grts mat
    grts = build_grts_mat(obs, exps, srcs)

    # Match to same srcs between acts and grts
    grts = match(srcs, grts.columns, grts.T.values).T

    # Build msks tensor
    msks, grpbys, grps = build_msks_tensor(obs, groupby)

    return acts, grts, msks, srcs, mthds, grpbys, grps


def rename_obs(obs, perturb, sign):
    # Check if names are in columns
    if perturb not in obs.columns:
        msg = (
            f"Column name '{perturb}' not found in obs. "
            f"Please specify a valid column."
        )
        raise ValueError(msg)

    # Check that they are not the same
    if perturb == sign:
        msg = f"perturb={perturb} and sign={sign} " f"cannot have the same value."
        raise ValueError(msg)

    # Validate sign
    if isinstance(sign, str):
        if sign not in obs.columns:
            msg = (
                f"Column name '{sign}' not found in obs. "
                f"Please specify a valid column."
            )
        unq = np.sort(np.unique(obs[sign].values))
        lbl = np.array([-1, 1])
        if not np.all(np.isin(unq, lbl)):
            msg = f"`sign` values can only be -1 or 1, got {list(unq)}."
            raise ValueError(msg)
    elif sign in (1, -1):
        obs = obs.copy()
        obs["sign"] = sign
        sign = "sign"
    else:
        msg = "If sign is not a column name, it must be 1 or -1."
        raise ValueError(msg)

    # Rename
    obs = obs.rename(columns={perturb: "perturb", sign: "sign"})

    return obs


def check_groupby(obs, groupby, perturb, by):
    if groupby is not None:
        if type(groupby) is str:
            groupby = [groupby]

        for grp_i in groupby:
            i = [grp_i] if isinstance(grp_i, str) else grp_i
            # For each group inside each groupby
            for j in i:
                # Check if perturb is in groupby when by=source
                if perturb == j and by == "source":
                    msg = (
                        f"'{perturb=}' column cannot be in groupby if by='source'."
                        f"Please remove it."
                    )
                    raise ValueError(msg)

                # Assert that columns exist in obs
                if j not in obs.columns:
                    msg = (
                        f"Column name '{j}' not found in obs. "
                        f"Please specify a valid column."
                    )
                    raise ValueError(msg)

                # Assert that column doesn't contain "|"
                if "|" in j:
                    msg = (
                        f"Column names cannot contain the character '|', "
                        f"please rename column {j}."
                    )
                    raise ValueError(msg)

    return groupby
