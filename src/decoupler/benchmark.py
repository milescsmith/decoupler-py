"""
Functions to benchmark methods and nets.
Functions to benchmark methods and nets using perturbation experiments.
"""

import numpy as np
import pandas as pd

from decoupler.decouple import decouple
from decoupler.pre import filt_min_n, rename_net
from decoupler.utils_anndata import extract_psbulk_inputs
from decoupler.utils_benchmark import (
    adjust_sign,
    append_metrics_scores,
    check_groupby,
    format_acts_grts,
    rename_obs,
    validate_metrics,
)


def get_performances(
    res,
    obs,
    groupby,
    by,
    metrics,
    use_pval=None,
    min_exp=5,
    pi0=0.5,
    n_iter=1000,
    seed=42,
    verbose=False,
):
    # Return acts, grts and msks tensors
    acts, grts, msks, srcs, mthds, grpbys, grps = format_acts_grts(
        res, obs, groupby, use_pval
    )

    # Init empty df
    df = []
    if msks is not None:
        n_grpbys = len(msks)
        for i in range(n_grpbys):
            msk_i = msks[i]
            grpby_i = grpbys[i]
            grps_i = grps[i]
            n_grps = len(grps_i)
            if verbose:
                pass
            for j in range(n_grps):
                msk = msk_i[j]
                grp = grps_i[j]
                n = np.sum(msk)

                # If enough exps, subset by group
                if n >= min_exp:
                    act, grt = acts[msk, :, :], grts[msk, :]

                    # Special case when groupby == perturb, remove extra grts
                    if grp in srcs:
                        m = grp == srcs
                        grt[:, ~m] = 0.0

                    # Compute and append scores to df
                    append_metrics_scores(
                        df,
                        grpby_i,
                        grp,
                        act,
                        grt,
                        srcs,
                        mthds,
                        metrics,
                        by,
                        min_exp=min_exp,
                        pi0=pi0,
                        n_iter=n_iter,
                        seed=seed,
                    )
    else:
        n_exp = acts.shape[0]
        if n_exp >= min_exp:
            # Compute and append scores to df
            if verbose:
                pass
            append_metrics_scores(
                df,
                None,
                None,
                acts,
                grts,
                srcs,
                mthds,
                metrics,
                by,
                min_exp=min_exp,
                pi0=pi0,
                n_iter=n_iter,
                seed=seed,
            )

    # Format df
    df = pd.DataFrame(
        df, columns=["groupby", "group", "source", "method", "metric", "score", "ci"]
    )

    return df


def format_benchmark_inputs(
    mat,
    obs,
    perturb,
    sign,
    net,
    groupby,
    by,
    f_expr=True,
    f_srcs=False,
    source="source",
    target="target",
    weight="weight",
    min_n=5,
    verbose=False,
    use_raw=True,
    decouple_kws=None,
):
    # Extract inputs
    if decouple_kws is None:
        decouple_kws = {}
    if verbose:
        pass
    mat, obs, var = extract_psbulk_inputs(mat, obs, layer=None, use_raw=use_raw)

    # Format groupby
    groupby = check_groupby(obs, groupby, perturb, by)

    # Rename obs
    obs = rename_obs(obs, perturb, sign)

    # Rename net
    if verbose:
        pass
    net = rename_net(
        net,
        source=decouple_kws["source"],
        target=decouple_kws["target"],
        weight=decouple_kws["weight"],
    )
    net = filt_min_n(var.index.values.astype("U"), net, min_n=decouple_kws["min_n"])

    # Remove experiments without sources in net
    if f_expr:
        msk = np.full((obs["perturb"].size,), False)
        srcs = net["source"].values.astype("U")
        for i, src in enumerate(obs["perturb"]):
            msk[i] = np.any(np.isin(src, srcs))
        mat, obs = mat[msk], obs.loc[msk]
        if verbose:
            np.sum(~msk)

    # Remove sources without experiments in obs
    if f_srcs:
        srcs = net.loc[:, "source"].values
        msk = np.isin(srcs, obs["perturb"].values.ravel())
        net = net.loc[msk]
        if verbose:
            np.unique(srcs[~msk]).size

    return mat, obs, var, net, groupby


def _benchmark(
    mat,
    obs,
    net,
    perturb,
    sign,
    metrics=None,
    groupby=None,
    by="experiment",
    f_expr=True,
    f_srcs=False,
    use_pval=None,
    min_exp=5,
    pi0=0.5,
    n_iter=1000,
    seed=42,
    verbose=True,
    use_raw=True,
    decouple_kws=None,
):
    # Format inputs
    if decouple_kws is None:
        decouple_kws = {}
    if metrics is None:
        metrics = ["auroc", "auprc"]
    mat, obs, var, net, groupby = format_benchmark_inputs(
        mat,
        obs,
        perturb,
        sign,
        net,
        groupby,
        by,
        f_expr=f_expr,
        f_srcs=f_srcs,
        verbose=verbose,
        use_raw=use_raw,
        decouple_kws=decouple_kws,
    )

    # Adjust sign
    mat = adjust_sign(mat, obs["sign"].values)
    obs["sign"] = 1

    # Reset net names args
    decouple_kws["source"] = "source"
    decouple_kws["target"] = "target"
    decouple_kws["weight"] = "weight"

    # Run prediction
    if verbose:
        srcs = []
        for p in obs.loc[:, "perturb"]:
            if isinstance(p, list):
                srcs.extend(p)
            else:
                srcs.append(p)
        np.unique(srcs).size
    res = decouple([mat, obs.index, var.index], net, verbose=verbose, **decouple_kws)

    # Compute metrics
    if verbose:
        pass
    df = get_performances(
        res,
        obs,
        groupby,
        by,
        metrics,
        use_pval=use_pval,
        min_exp=min_exp,
        pi0=pi0,
        n_iter=n_iter,
        seed=seed,
        verbose=verbose,
    )
    if verbose:
        pass

    return df


def benchmark(
    mat,
    obs,
    net,
    perturb,
    sign,
    metrics=None,
    groupby=None,
    by="experiment",
    f_expr=True,
    f_srcs=False,
    use_pval=None,
    min_exp=5,
    pi0=0.5,
    n_iter=1000,
    seed=42,
    verbose=True,
    use_raw=True,
    decouple_kws=None,
):
    """
    Benchmark methods or networks on a given set of perturbation experiments using activity inference with decoupler.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    obs : DataFrame or None
        Metadata containing the perturbed targets and the sign of the perturbation. If mat is AnnData, use mat.obs
        attribute instead.
    net : DataFrame, dict
        Network in long format. Can be dictionary of nets, where key is the name and value is the long format DataFrame.
    perturb : str
        Column name in obs with perturbed sources.
    sign : str, int
        Column name in obs with sign of the perturbation. Can be set to 1 or -1 if all experiments are overexpression or
        knockouts, respectively.
    metrics : list, str
        Performance metric(s) to compute. See the description of get_performance for more details.
    groupby : list, str, None
        Performance metrics(s) can be computed per groups if enough experiments are available.
    by : str
        Whether to evaluate performances at the "experiment" or at the "source" level.
    f_expr : bool
        Whether to filter out experiments whose perturbed sources are not in the given net. Defaults to True.
    f_srcs : bool
        Whether to fitler out sources in net for which there are not perturbation data. Defaults to False.
    use_pval: None, float
        Whether to fitler out activity scores by a given p-value after FDR correction (BH). Defaults to None. Methods that
        generate no p-value will use the (1 - pvalue) quantile based on activity score as being significant.
    min_exp : int
        Minimum of perturbation experiments per group.
    pi0 : float
        Reference ratio for calibrated metrics. Corresponds to the baseline/reference class inbalance to which
        to set the metric.
    n_iter : int
        Number of downsampling iterations used for the 'mcroc' and 'mcprc' metrics.
    seed : int
        Random seed to use.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.
    decouple_kws : dict
        Parameters for the decoupler.decouple function. If more than one net, use a nested dictionary where the main
        key is the network name and the value is a dictionary with the requiered arguments.

    Returns
    -------
    df : DataFrame
        DataFrame containing the metrics' scores.
    """

    # Init default args
    if decouple_kws is None:
        decouple_kws = {}
    if metrics is None:
        metrics = ["auroc", "auprc", "mcauroc", "mcauprc", "rank", "nrank", "recall"]
    default_kws = {
        "source": "source",
        "target": "target",
        "weight": "weight",
        "min_n": 5,
    }

    # Validate by
    if by not in ["experiment", "source"]:
        msg = 'Argument `by` has to be either "experiment" or "source".'
        raise ValueError(msg)

    # Validate metrics
    validate_metrics(metrics)

    # Validate pi0
    if pi0 is not None:
        if pi0 < 0 or pi0 > 1:
            msg = "Argument `pi0` needs to be between 0 and 1."
            raise ValueError(msg)

    # Run benchmark per net
    if not isinstance(net, dict):
        # Update decouple args
        decouple_kws = {**default_kws, **decouple_kws}

        # Run benchmark
        df = _benchmark(
            mat,
            obs,
            net,
            perturb,
            sign,
            metrics,
            groupby,
            by,
            f_expr,
            f_srcs,
            use_pval,
            min_exp,
            pi0,
            n_iter,
            seed,
            verbose,
            use_raw,
            decouple_kws,
        )
    else:
        df = []
        for net_name in net:
            if verbose:
                pass

            # Update decouple args
            decouple_kws.setdefault(net_name, {})
            decouple_kws[net_name] = {**default_kws, **decouple_kws[net_name]}

            # Run benchmark
            tmp = _benchmark(
                mat,
                obs,
                net[net_name],
                perturb,
                sign,
                metrics,
                groupby,
                by,
                f_expr,
                f_srcs,
                use_pval,
                min_exp,
                pi0,
                n_iter,
                seed,
                verbose,
                use_raw,
                decouple_kws[net_name],
            )
            tmp["net"] = net_name
            df.append(tmp)

        # Merge all results
        df = pd.concat(df)

    return df
