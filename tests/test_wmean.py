import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from decoupler.method_wmean import run_wmean, wmean


def test_wmean():
    m = csr_matrix(
        np.array(
            [
                [7.0, 1.0, 1.0, 1.0],
                [4.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 5.0, 1.0],
                [1.0, 1.0, 6.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    net = np.array([[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]], dtype=np.float32)
    est, norm, corr, pvl = wmean(m, net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))
    est, norm, corr, pvl = wmean(m.toarray(), net, 1000, 10000, 42, True)
    assert norm[0, 0] > 0
    assert norm[1, 0] > 0
    assert norm[2, 0] < 0
    assert norm[3, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))


def test_run_wmean():
    m = np.array(
        [
            [7.0, 1.0, 1.0, 1.0],
            [4.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, 6.0, 2.0],
        ]
    )
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3", "G4"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame(
        [["T1", "G1", 1], ["T1", "G2", 2], ["T2", "G3", -3], ["T2", "G4", 4]],
        columns=["source", "target", "weight"],
    )
    run_wmean(adata, net, verbose=True, use_raw=False, min_n=0, times=2)
