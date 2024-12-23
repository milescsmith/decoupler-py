import numpy as np
import pandas as pd
from anndata import AnnData

from decoupler.method_zscore import run_zscore, zscore


def test_zscore():
    m = np.array(
        [
            [-7.0, -1.0, 1.0, 1.0],
            [-4.0, -2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, 6.0, 2.0],
            [-8.0, -7.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    net = np.array([[1.0, 0.0], [1, 0.0], [0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
    act, pvl = zscore(m, net)
    assert act[0, 0] < 0
    assert act[1, 0] < 0
    assert act[2, 0] > 0
    assert act[3, 0] > 0
    assert act[4, 0] < 0
    assert np.all((0.0 <= pvl) * (pvl <= 1.0))

    act2, pvl2 = zscore(m, net, flavor="KSEA")
    assert act2[0, 0] < 0
    assert act2[1, 0] < 0
    assert act2[2, 0] < 0
    assert act2[3, 0] < 0
    assert act2[4, 0] < 0
    assert np.all((0.0 <= pvl2) * (pvl2 <= 1.0))


def test_run_zscore():
    m = np.array(
        [
            [-7.0, -1.0, 1.0, 1.0],
            [-4.0, -2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, -6.0, -8.0],
            [-8.0, -7.0, 1.0, 1.0],
        ]
    )
    r = np.array(["S1", "S2", "S3", "S4", "S5"])
    c = np.array(["G1", "G2", "G3", "G4"])
    df = pd.DataFrame(m, index=r, columns=c)
    net = pd.DataFrame(
        [["T1", "G1", 1], ["T1", "G2", 1], ["T2", "G3", -1], ["T2", "G4", -1]],
        columns=["source", "target", "weight"],
    )
    res = run_zscore(df, net, verbose=True, use_raw=False, min_n=0)
    assert res[0].loc["S1", "T2"] < 0
    assert res[0].loc["S2", "T2"] < 0
    assert res[0].loc["S3", "T2"] < 0
    assert res[0].loc["S4", "T2"] > 0
    assert res[0].loc["S5", "T2"] < 0
    assert res[1].map(lambda x: 0 <= x <= 1).all().all()

    res2 = run_zscore(df, net, verbose=True, use_raw=False, min_n=0, flavor="KSEA")
    assert res2[0].loc["S1", "T2"] > 0
    assert res2[0].loc["S2", "T2"] < 0
    assert res2[0].loc["S3", "T2"] < 0
    assert res2[0].loc["S4", "T2"] > 0
    assert res2[0].loc["S5", "T2"] > 0
    assert res2[1].map(lambda x: 0 <= x <= 1).all().all()

    adata = AnnData(df.astype(np.float32))
    run_zscore(adata, net, verbose=True, use_raw=False, min_n=0)
