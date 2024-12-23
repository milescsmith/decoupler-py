import numpy as np
import pandas as pd

from decoupler.consensus import cons, mean_z_scores, z_score


def test_z_score():
    arr = np.array([1.0, 2.0, 6.0], dtype=np.float32)
    z_score(arr)


def test_mean_z_scores():
    arr = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]],
        dtype=np.float32,
    )
    mean_z_scores(arr)


def test_cons():
    mlm_estimate = pd.DataFrame(
        [[3.5, -0.5, 0.3], [3.6, -0.6, 0.04], [-1, 2, -1.8]],
        columns=["T1", "T2", "T3"],
        index=["C1", "C2", "C3"],
    )
    mlm_estimate.name = "mlm_estimate"
    mlm_pvals = pd.DataFrame(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        columns=["T1", "T2", "T3"],
        index=["C1", "C2", "C3"],
    )
    mlm_pvals.name = "mlm_pvals"
    ulm_estimate = pd.DataFrame(
        [[3.9, -0.2, 0.8], [3.2, -0.1, 0.09], [-2, 3, -2.3]],
        columns=["T1", "T2", "T3"],
        index=["C1", "C2", "C3"],
    )
    ulm_estimate.name = "ulm_estimate"
    ulm_pvals = pd.DataFrame(
        [[0.2, 0.1, 0.3], [0.5, 0.3, 0.2], [0.4, 0.5, 0.3]],
        columns=["T1", "T2", "T3"],
        index=["C1", "C2", "C3"],
    )
    ulm_pvals.name = "ulm_pvals"
    res = {
        mlm_estimate.name: mlm_estimate,
        mlm_pvals.name: mlm_pvals,
        ulm_estimate.name: ulm_estimate,
        ulm_pvals.name: ulm_pvals,
    }
    cons(res)
