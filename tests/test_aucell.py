import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from decoupler.method_aucell import aucell, run_aucell


def test_aucell():
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
    net = pd.Series(
        [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)],
        index=["T1", "T2"],
    )
    n_up = np.array([4], dtype=np.int64)[0]
    aucell(m, net, n_up, False)

    act = aucell(m, net, n_up, False)
    assert act[0, 0] > 0.7
    assert act[1, 0] > 0.7
    assert act[2, 0] < 0.7
    assert act[3, 0] < 0.7
    assert np.all((0.0 <= act) * (act <= 1.0))
    act = aucell(m.toarray(), net, n_up, False)
    assert act[0, 0] > 0.7
    assert act[1, 0] > 0.7
    assert act[2, 0] < 0.7
    assert act[3, 0] < 0.7
    assert np.all((0.0 <= act) * (act <= 1.0))


def test_run_aucell():
    m = np.array([[7.0, 1.0, 1.0], [4.0, 2.0, 1.0], [1.0, 2.0, 5.0], [1.0, 1.0, 6.0]])
    r = np.array(["S1", "S2", "S3", "S4"])
    c = np.array(["G1", "G2", "G3"])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df.astype(np.float32))
    net = pd.DataFrame(
        [["T1", "G2"], ["T1", "G4"], ["T2", "G3"], ["T2", "G1"]],
        columns=["source", "target"],
    )
    run_aucell(adata, net, n_up=2, min_n=0, verbose=True, use_raw=False)
    with pytest.raises(ValueError):
        run_aucell(adata, net, n_up=-3, min_n=0, verbose=True, use_raw=False)
