import numpy as np
import pandas as pd
import pytest
import sklearn as sk
from anndata import AnnData
from scipy.sparse import csr_matrix

from decoupler.method_udt import fit_dt, run_udt, udt


@pytest.fixture
def udt_net():
    return np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.7, 0.0],
            [0.0, 1.0],
            [0.0, -0.5],
            [0.0, -1.0],
        ]
    )


@pytest.fixture
def fit_dt_sample():
    return np.array([7.0, 6.0, 1.0, -3.0, -4.0, 0.0])


def test_fit_dt(udt_net, fit_dt_sample):
    a = fit_dt(sk, udt_net[:, 0], fit_dt_sample, seed=42, min_leaf=1)
    b = fit_dt(sk, udt_net[:, 1], fit_dt_sample, seed=42, min_leaf=1)
    assert a > b


@pytest.fixture
def udt_m():
    return csr_matrix(np.array([[7.0, 6.0, 1.0, -3.0, -4.0, 0.0]]))


def test_udt_sparse(udt_m, udt_net):
    a, b = udt(udt_m, udt_net, seed=42, min_leaf=1)[0]
    assert a > b


def test_udt_dense(udt_m, udt_net):
    a, b = udt(udt_m.toarray(), udt_net, seed=42, min_leaf=1)[0]
    assert a > b


@pytest.fixture
def mdt_m():
    return np.array(
        [[7.0, 1.0, 1.0], [4.0, 2.0, 1.0], [1.0, 2.0, 5.0], [1.0, 1.0, 6.0]]
    )


@pytest.fixture
def mdt_r():
    return np.array(["S1", "S2", "S3", "S4"])


@pytest.fixture
def mdt_c():
    return np.array(["G1", "G2", "G3"])


@pytest.fixture
def mdt_df(mdt_m, mdt_r, mdt_c):
    return pd.DataFrame(mdt_m, index=mdt_r, columns=mdt_c)


@pytest.fixture
def mdt_adata(mdt_df):
    return AnnData(mdt_df.astype(np.float32))


@pytest.fixture
def mdt_net():
    return pd.DataFrame(
        [["T1", "G2", 1], ["T1", "G4", 2], ["T2", "G3", 3], ["T2", "G1", 1]],
        columns=["source", "target", "weight"],
    )


def test_run_mdt(mdt_adata, mdt_net):
    run_udt(mdt_adata, mdt_net, verbose=True, use_raw=False, min_n=0)
