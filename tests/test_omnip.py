import pandas as pd
import pytest

from decoupler.omnip import (
    get_collectri,
    get_dorothea,
    get_ksn_omnipath,
    get_progeny,
    get_resource,
    merge_genes_to_complexes,
    show_resources,
)


def test_get_resource():
    res = get_resource("TFcensus")
    assert type(res) is pd.DataFrame
    assert res.shape[0] > 0


def test_show_resources():
    lst = show_resources()
    assert type(lst) is list
    assert len(lst) > 0


def test_get_dorothea():
    df = get_dorothea(organism="human")
    assert type(df) is pd.DataFrame
    assert df.shape[0] > 0
    with pytest.raises(AssertionError):
        get_dorothea(organism="asdfgh")
    get_dorothea(organism="mouse")


def test_():
    df = pd.DataFrame()
    df["source_genesymbol"] = ["JUN1", "JUN2", "RELA", "NFKB3", "STAT1"]
    merge_genes_to_complexes(df)
    assert df["source_genesymbol"].unique().size == 3


def test_get_collectri():
    df = get_collectri(organism="human", split_complexes=False)
    assert type(df) is pd.DataFrame
    assert df.shape[0] > 0
    with pytest.raises(AssertionError):
        get_collectri(organism="asdfgh", split_complexes=False)
    get_collectri(organism="mouse", split_complexes=False)
    subunits_df = get_collectri(organism="human", split_complexes=True)
    assert df.shape[0] < subunits_df.shape[0]


def test_get_progeny():
    df = get_progeny(organism="human", top=100)
    n_paths = len(df["source"].unique())
    n_rows = n_paths * 100
    assert type(df) is pd.DataFrame
    assert df.shape[0] == n_rows
    with pytest.raises(AssertionError):
        get_progeny(organism="asdfgh")


def test_get_ksn_omnipath():
    df = get_ksn_omnipath()
    assert type(df) is pd.DataFrame
    assert df.shape[0] > 0
