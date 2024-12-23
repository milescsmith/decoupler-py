import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from decoupler.method_viper import (
    aREA,
    get_inter_pvals,
    run_viper,
    shadow_regulon,
    viper,
)


def test_get_inter_pvals():
    nes_i = np.array([1, 2, 3, 4])
    ss_i = np.array([5, 4, 3, 2])
    sub_net = np.array(
        [[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]], dtype=np.float32
    )
    n_targets = 2
    get_inter_pvals(nes_i, ss_i, sub_net, n_targets)


def test_shadow_regulon():
    nes_i = np.array([1, 2])
    ss_i = np.array([5, 4])
    net = np.array([[1.0, 0.0], [2, 2.0], [1.0, -3.0], [1.0, 4.0]], dtype=np.float32)
    shadow_regulon(nes_i, ss_i, net, reg_sign=1.96, n_targets=2, penalty=20)


def test_aREA():
    m = np.array(
        [
            [7.0, 1.0, 1.0, 1.0],
            [4.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 5.0, 1.0],
            [1.0, 1.0, 6.0, 2.0],
        ]
    )
    net = np.array([[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]])
    aREA(m, net)


def test_viper():
    m = csr_matrix(
        np.array(
            [
                [7.0, 1.0, 1.0, 1.0],
                [4.0, 2.0, 1.0, 2.0],
                [1.0, 2.0, 5.0, 1.0],
                [1.0, 1.0, 6.0, 2.0],
            ]
        )
    )
    net = np.array([[1.0, 0.0], [2, 0.0], [0.0, -3.0], [0.0, 4.0]])
    viper(m, net, pleiotropy=True, reg_sign=0.95, n_targets=1, penalty=20, verbose=True)


def test_run_viper():
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
    run_viper(adata, net, verbose=True, use_raw=False, min_n=0)
