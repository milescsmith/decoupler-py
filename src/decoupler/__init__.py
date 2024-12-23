from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from decoupler.benchmark import (
    benchmark,
    format_benchmark_inputs,
    get_performances,
)
from decoupler.consensus import cons
from decoupler.decouple import decouple, run_consensus
from decoupler.method_aucell import run_aucell
from decoupler.method_gsea import get_gsea_df, run_gsea
from decoupler.method_gsva import run_gsva
from decoupler.method_mdt import run_mdt
from decoupler.method_mlm import run_mlm
from decoupler.method_ora import get_ora_df, run_ora, test1r
from decoupler.method_udt import run_udt
from decoupler.method_ulm import run_ulm
from decoupler.method_viper import run_viper
from decoupler.method_wmean import run_wmean
from decoupler.method_wsum import run_wsum
from decoupler.method_zscore import run_zscore
from decoupler.metrics import (
    metric_auprc,
    metric_auroc,
    metric_mcauprc,
    metric_mcauroc,
    metric_nrank,
    metric_rank,
    metric_recall,
)
from decoupler.omnip import (
    get_collectri,
    get_dorothea,
    get_ksn_omnipath,
    get_progeny,
    get_resource,
    show_resources,
    translate_net,
)
from decoupler.plotting import (
    plot_associations,
    plot_barplot,
    plot_barplot_df,
    plot_dotplot,
    plot_filter_by_expr,
    plot_filter_by_prop,
    plot_metrics_boxplot,
    plot_metrics_scatter,
    plot_metrics_scatter_cols,
    plot_network,
    plot_psbulk_samples,
    plot_running_score,
    plot_targets,
    plot_violins,
    plot_volcano,
    plot_volcano_df,
)
from decoupler.pre import (
    break_ties,
    extract,
    filt_min_n,
    get_net_mat,
    mask_features,
    match,
    rename_net,
    return_data,
)
from decoupler.utils import (
    assign_groups,
    check_corr,
    dense_run,
    get_toy_data,
    melt,
    p_adjust_fdr,
    read_gmt,
    show_methods,
    shuffle_net,
    summarize_acts,
)
from decoupler.utils_anndata import (
    filter_by_expr,
    filter_by_prop,
    format_contrast_results,
    get_acts,
    get_contrast,
    get_metadata_associations,
    get_pseudobulk,
    get_top_targets,
    rank_sources_groups,
    swap_layer,
)
from decoupler.utils_benchmark import get_toy_benchmark_data, show_metrics

__all__ = [
    "assign_groups",
    "benchmark",
    "break_ties",
    "check_corr",
    "cons",
    "decouple",
    "dense_run",
    "extract",
    "filt_min_n",
    "filter_by_expr",
    "filter_by_prop",
    "format_benchmark_inputs",
    "format_contrast_results",
    "get_acts",
    "get_collectri",
    "get_contrast",
    "get_dorothea",
    "get_gsea_df",
    "get_ksn_omnipath",
    "get_metadata_associations",
    "get_net_mat",
    "get_ora_df",
    "get_performances",
    "get_progeny",
    "get_pseudobulk",
    "get_resource",
    "get_top_targets",
    "get_toy_benchmark_data",
    "get_toy_data",
    "mask_features",
    "match",
    "melt",
    "metric_auprc",
    "metric_auroc",
    "metric_mcauprc",
    "metric_mcauroc",
    "metric_nrank",
    "metric_rank",
    "metric_recall",
    "p_adjust_fdr",
    "plot_associations",
    "plot_barplot",
    "plot_barplot_df",
    "plot_dotplot",
    "plot_filter_by_expr",
    "plot_filter_by_prop",
    "plot_metrics_boxplot",
    "plot_metrics_scatter",
    "plot_metrics_scatter_cols",
    "plot_network",
    "plot_psbulk_samples",
    "plot_running_score",
    "plot_targets",
    "plot_violins",
    "plot_volcano",
    "plot_volcano_df",
    "rank_sources_groups",
    "read_gmt",
    "rename_net",
    "return_data",
    "run_aucell",
    "run_consensus",
    "run_gsea",
    "run_gsva",
    "run_mdt",
    "run_mlm",
    "run_ora",
    "run_udt",
    "run_ulm",
    "run_viper",
    "run_wmean",
    "run_wsum",
    "run_zscore",
    "show_methods",
    "show_metrics",
    "show_resources",
    "shuffle_net",
    "summarize_acts",
    "swap_layer",
    "test1r",
    "translate_net",
]
