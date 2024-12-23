import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from decoupler.method_mdt import check_if_skranger, fit_rf, mdt, run_mdt


def test_check_if_skranger():
    sr = check_if_skranger()
    assert sr is not None


def test_fit_rf():
    net = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.7, 0.0],
            [0.0, 1.0],
            [0.0, -0.5],
            [0.0, -1.0],
        ]
    )
    sample = np.array([7.0, 6.0, 1.0, -3.0, -4.0, 0.0])
    sr = check_if_skranger()
    a, b = fit_rf(sr, net, sample, min_leaf=2)
    assert a > b


def test_mdt():
    m = csr_matrix(np.array([[7.0, 6.0, 1.0, -3.0, -4.0, 0.0]]))
    net = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.7, 0.0],
            [0.0, 1.0],
            [0.0, -0.5],
            [0.0, -1.0],
        ]
    )
    a, b = mdt(m, net, seed=42, trees=100, min_leaf=2)[0]
    assert a > b
    a, b = mdt(m.toarray(), net, seed=42, trees=100, min_leaf=2)[0]
    assert a > b


def test_run_mdt():
    m = np.array([[7.0, 1.0, 1.0], [4.0, 2.0, 1.0], [1.0, 2.0, 5.0], [1.0, 1.0, 6.0]])
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame(
        [["T1", "G2", 1], ["T1", "G4", 2], ["T2", "G3", 3], ["T2", "G1", 1]],
        columns=["source", "target", "weight"],
    )
    run_mdt(adata, net, verbose=True, use_raw=False, min_n=0)
