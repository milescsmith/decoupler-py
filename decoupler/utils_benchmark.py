"""
Utility functions to benchmark resources on known data
"""

import numpy as np
import pandas as pd
import decoupler as dc

from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_scores_GT(nexp=50, ncol = 4):
    """
    Generate random scores and groud-truth matrix, for testing

    Args:
        nexp (int, optional): Number of rows/experiments. Defaults to 50.
        ncol (int, optional): Number of classes/TFs/pathways. Defaults to 4.

    Returns:
        _type_: (DataFrame): Dataframe with scores and associated ground truth
    """
    df = np.random.randn(nexp, ncol)
    ind = np.random.randint(0, df.shape[1] ,size=df.shape[0])
    gt = np.zeros(df.shape)
    gt[range(gt.shape[0]),ind] = 1

    return pd.DataFrame(np.column_stack((df.flatten(), gt.flatten())), columns = ['score', 'GT'])

"""
Downsample ground truth vector 
"""

def down_sampling(y, seed=7, n = 100):
    """
    Downsampling of ground truth
    
    Parameters
    ----------
    
    y: array
        binary groundtruth vector 
        
    seed: arbitrary seed for random sampling
    
    n: number of iterations
        
    Returns
    -------
    ds_msk: list of downsampling masks for input vectors
    """
    
    msk = []
    rng = default_rng(seed)
    
    # Downsampling
    for i in range(n):
        tn_index = np.where(y == 0)[0]
        tp_index = np.where(y != 0)[0]
        ds_array = rng.choice(tn_index, size=len(tp_index), replace=True)
        ds_msk = np.hstack([ds_array, tp_index])

        msk.append(ds_msk)

    return msk

"""
Compute AUC of ROC or PRC
"""

def get_auc(x, y, mode):
    """
    Computes AUROC for each label
    
    Parameters
    ----------
    
    x: array
        binary groundtruth vector
        
    y: array (flattened)
        vector of continuous values
        
        
    Returns
    -------
    auc: value of auc
    """

    if mode == "roc":
        auc = roc_auc_score(x, y) 
    elif mode == "prc":
        auc = average_precision_score(x, y)
    else: 
        raise ValueError("mode can only be roc or prc")
    return auc

def get_target_masks(long_data, targets, subset = None):
    """
    Generates a list of indices of a DataFrame that correspond to prediction scores and associated ground-truth for each target

    Args:
        long_data (DataFrame): DataFrame with a 'score' and 'GT' column.
        targets (list): List of targets (have to be in correct order) for which there are entries in long_data.
        subset (list, optional): A subset of the targets for which to make masks. If None, then the masks will be made for all targets. Defaults to None.

    Returns:
        target_ind: List of data indices for each target in the targets object
        target_names : Target name corresponding to the elements in target_ind. Elements in subset are filtered and put in same order as in targets.
    """

    if long_data.shape[0] < len(targets):
        raise ValueError('The data given is smaller than the number of targets')
    elif long_data.shape[0] % len(targets) != 0:
        raise ValueError('The data is likely misshapen: the number of rows cannot be divided by the number of targets')

    if subset is not None:
        iterateover = np.argwhere(np.in1d(targets, subset)).flatten().tolist()
        target_names = [targets[i] for i in iterateover]
    else:
        iterateover = range(len(targets))
        target_names = targets

    target_ind = []
    for target in iterateover:
        target_ind.append(np.arange(target, long_data.shape[0], len(targets)))

    return target_ind, target_names

def get_performance(data, metric = 'mcroc', n_iter = 100, seed = 42, method_name = None):
    """
    Compute binary classifier performance

    Args:
        data (DataFrame): DataFrame with a 'score' and 'GT' column. 'GT' column contains groud-truth ( e.g. 0, 1). 'score' can be continuous or same as 'GT'
        metric (str or list of str, optional): Which metric(s) to use. Currently implemeted methods are: 'mcroc', 'mcprc', 'roc', 'prc'. Defaults to 'mcroc'.
        n_iter (int, optional): Number of iterations for the undersampling procedures for the 'mcroc' and 'mcprc' metrics. Defaults to 100.
        seed (int, optional): Seed used to generate the undersampling for the 'mcroc' and 'mcprc' metrics. Defaults to 42.
        method_name (str, optional): Name of the decoupler method used to do activity prediction. Added as prefix to the output dictionary: e.g. 'mlm_roc'. Defaults to 100.

    Returns:
        perf: Dict of prediction performance(s) on the given data. 'mcroc' and 'mcprc' metrics will return the values for each sampling. Other methods return a single value.
    """

    available_metrics = ['mcroc', 'mcprc', 'roc', 'prc']
    metrics = [available_metrics[i] for i in np.argwhere(np.in1d(available_metrics, metric)).flatten().tolist()]

    if len(metrics) == 0:
        raise ValueError('None of the performance metrics given as parameter have been implemented')

    if 'mcroc' in metrics or 'mcprc' in metrics:
        masks = down_sampling(y = data['GT'].values, seed=seed, n=n_iter)

    perf = {}
    for met in metrics:
        if met == 'mcroc' or met == 'mcprc':
            # Compute AUC for each mask (with equalised class priors)
            aucs = []
            for mask in tqdm(masks):
                auc = get_auc(x = data['GT'][mask],
                            y = data['score'][mask],
                            mode = met[2:])
                aucs.append(auc)

        elif met == 'roc' or met == 'prc':
            # Compute AUC on the whole (unequalised class priors) data
            aucs = get_auc(x = data['GT'], y = data['score'], mode = met)
        
        if method_name is None:
            perf[met] = aucs
        else:
            perf[method_name + '_' + met] = aucs

    return perf

def get_target_performance(data, targets, metric='mcroc', subset = None, n_iter = 100, seed = 42):
    """
    Compute binary classifier performance for each target or subet of targets

    Args:
        data (DataFrame): DataFrame with a 'score' and 'GT' column. 'GT' column contains groud-truth ( e.g. 0, 1). 'score' can be continuous or same as 'GT'
        targets (list of str): List of targets (have to be in correct order) for which there are entries in data.
        metric (str, or list of str optional): Which metrics to use. Currently implemeted methods are: 'mcroc', 'mcprc', 'roc', 'prc'. Defaults to 'mcroc'.
        subset (list of str, optional): A subset of the targets for which to compute performance. If None, then the performance will be calculated for all targets. Defaults to None.
        n_iter (int, optional): Number of iterations for the undersampling procedures for the 'mcroc' and 'mcprc' metrics. Defaults to 100.
        seed (int, optional):  Seed used to generate the undersampling for the 'mcroc' and 'mcprc' metrics. Defaults to 42.

    Returns:
        perf : Dict of prediction performance(s) for each target or subset of targets. 'mcroc' and 'mcprc' metrics will return the values for each sampling. Other methods return a single value.
    """

    masks, target_names = get_target_masks(data, targets, subset = subset)

    perf = {}
    for trgt, name in zip(masks, target_names):
        perf[name] = get_performance(data.iloc[trgt.tolist()].reset_index(), metric, n_iter, seed)

    return perf

def get_scores_GT(decoupler_results, metadata, meta_target_col = 'target'):
    """

    Convert decouple output to flattenend vectors and combine with GT information

    Args:
        decoupler_results (dict): Output of decouple
        metadata (DataFrame): Metadata of the perturbation experiment containing the activated/inhibited targets and the sign of the perturbation
        meta_target_col (str, optional): Column name in the metadata with perturbation targets. Defaults to 'target'.

    Returns:
        scores_gt: dict of flattenend score,gt dataframes for each method
    """
    computed_methods = list(set([i.split('_')[0] for i in decoupler_results.keys()])) # get the methods that were able to be computed (filtering of methods done by decouple)
    scores_gt = {}
    for m in computed_methods:
        # estimates = res[m + 'estimate']
        # pvals = res[m + 'pvals']

        # remove experiments with no prediction for the perturbed TF
        missing = list(set( metadata[meta_target_col]) - set(decoupler_results[m + '_estimate'].columns))
        keep = [trgt not in missing for trgt in metadata[meta_target_col].to_list()]
        meta = metadata[keep]
        estimates = decoupler_results[m + '_estimate'][keep]
        # pvals = res[m + '_pvals'][keep]

        # mirror estimates
        estimates = estimates.mul(meta['sign'], axis = 0)
        gt = meta.pivot(columns = meta_target_col, values = 'sign').fillna(0)

        # add 0s in the ground-truth array for targets predicted by decoupler
        # for which there is no ground truth in the provided metadata (assumed 0)
        missing = list(set(estimates.columns) - set(meta[meta_target_col]))
        gt = pd.concat([gt, pd.DataFrame(0, index= gt.index, columns=missing)], axis = 1, join = 'inner').sort_index(axis=1)

        # flatten and then combine estimates and GT vectors
        # set ground truth to be either 0 or 1
        df_scores = pd.DataFrame({'score': estimates.to_numpy().flatten(), 'GT': gt.to_numpy().flatten()})
        df_scores['GT'] = abs(df_scores['GT'])

        scores_gt[m] = df_scores

    return scores_gt

def run_benchmark(data, metadata, network, methods = None, metric = 'roc', meta_target_col = 'target', net_source_col = 'source', net_target_col = 'target', filter_experiments= True, filter_sources = False, **kwargs):
    """
    Benchmark methods or networks on a given set of perturbation experiments using activity inference with decoupler.

    Args:
        data (DataFrame): Gene expression data where each row is a perturbation experiment and each column a gene
        metadata (DataFrame): Metadata of the perturbation experiment containing the activated/inhibited targets and the sign of the perturbation
        network (DataFrame): Network in long format passed on to the decouple function
        methods (str or list of str, optional): List of methods to run. If none are provided use weighted top performers (mlm, ulm and wsum). To benchmark all methods set to "all". Defaults to None.
        metric (str or list of str, optional): Performance metric(s) to compute. See the description of get_performance for more details. Defaults to 'roc'.
        meta_target_col (str, optional): Column name in the metadata with perturbation targets. Defaults to 'target'.
        net_source_col (str, optional): Column name in network with source nodes. Defaults to 'source'.
        net_target_col (str, optional): Column name in net with target nodes. Defaults to 'target'.
        filter_experiments (bool, optional): Whether to filter out experiments whose perturbed targets cannot be infered from the given network. Defaults to True.
        filter_sources (bool, optional): Whether to fitler out sources in the network for which there are not perturbation experiments (reduces the number of predictions made by decouple). Defaults to False.
        **kwargs: other arguments to pass on to get_performance (e.g. n_iter etc)

    Returns:
        mean_perf: DataFrame containing the mean performance for each metric and for each method (mean has to be done for the mcroc and mcprc metrics)
        bench: dict containing the whole data for each method and metric. Useful if you want to see the distribution for each subsampling for the mcroc and mcprc methods
    """

    #subset by TFs with GT available
    if filter_sources:
        keep = [src in metadata[meta_target_col].to_list() for src in network[net_source_col].to_list()]
        network = network[keep]

    # filter out experiments without predictions available
    if filter_experiments:
        keep = [trgt in network[net_target_col].to_list() for trgt in metadata[meta_target_col].to_list()]
        data = data[keep]
        metadata = metadata[keep]

    # run prediction
    res = dc.decouple(data, network, methods=methods)

    scores_gt = get_scores_GT(res, metadata, meta_target_col)

    bench = {}
    for method in scores_gt.keys():
        print('Calculating performance metrics for', method)
        perf = get_performance(scores_gt[method], metric, method_name= method, **kwargs)
        bench.update(perf)
    
    #make dataframe with mean perfomances
    mean_perfs = {}
    for key, value in bench.items():
        mean_perfs[key]=np.mean(value)
    mean_perfs = pd.DataFrame.from_dict(mean_perfs, orient='index').reset_index(level=0)
    mean_perfs.columns = ['id','value']
    mean_perfs[['method','metric']] = mean_perfs['id'].str.split('_', expand=True)
    mean_perfs = mean_perfs.pivot(index='method', columns='metric', values='value')

    return mean_perfs, bench

def benchmark_scatterplot(mean_perf, x = 'mcroc', y = 'mcprc'):
    """
    Creates a scatter plot for each given method for two performance metrics

    Args:
        mean_perf (DataFrame): Mean performance of each method output by run_benchmark()
        x (str, optional): Which metric to plot on the x axis. Defaults to 'mcroc'.
        y (str, optional): Which metric to plot on the y axis. Defaults to 'mcprc'.

    Returns:
        ax: Axes of a scatter plot
    """

    ax = plt.subplot(111)
    ax.scatter(x = mean_perf[x], y = mean_perf[y])
    ax.set_aspect('equal')

    min_v = mean_perf[[x,y]].min().min()
    max_v = mean_perf[[x,y]].max().max()
    border = (max_v - min_v)/15

    ax.set_xlim(min_v - border, max_v + border)
    ax.set_ylim(min_v - border, max_v + border)

    if (x in ['roc','mcroc'] and y in ['roc','mcroc']) or (x in ['prc','mcprc'] and y in ['prc','mcprc']):
        ax.axline((0,0),slope=1, color = 'black', linestyle = ':')

    for i, label in enumerate(mean_perf.index):
        ax.annotate(label.capitalize(), (mean_perf[x][i], mean_perf[y][i]))

    if x in ['mcroc', 'mcprc', 'roc', 'prc']:
        x = x + ' AUC'

    if y in ['mcroc', 'mcprc', 'roc', 'prc']:
        y = y + ' AUC'

    ax.set_xlabel(x.upper())
    ax.set_ylabel(y.upper())

    return ax

def benchmark_boxplot(benchmark_data, metric = 'mcroc'):
    """
    Creates boxplots for an iterative performance metric (i.e. mcroc and mcprc)

    Args:
        benchmark_data (dict): dict containing complete output from run_benchmark()
        metric (str, optional): Metric to plot a distribution for. Either mcroc or mcprc. Defaults to 'mcroc'.

    Returns:
        ax: Axes of a boxplot
    """

    if not (metric == 'mcprc' or metric == 'mcroc'):
        raise ValueError('Plotting of boxplots only possible for the \'mcprc\' and \'mcroc\' methods')

    keys = [key for key in benchmark_data.keys() if metric in key.split('_')[1]]
    methods = [key.split('_')[0] for key in keys]

    if len(keys) == 0:
        raise ValueError('The given metric was not found in the benchmark data')

    fig = plt.figure()
    ax = plt.subplot(111)
    for i, key in enumerate(keys):
        ax.boxplot(benchmark_data[key], positions = [i])
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylabel(metric.upper() + ' AUC')
    ax.set_xticklabels([m.capitalize() for m in methods])
    
    return ax