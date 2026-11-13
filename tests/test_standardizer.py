import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn import clone
from sklearn.pipeline import make_pipeline

from scikit_mol.conversions import SmilesToMolTransformer
from scikit_mol.core import DEFAULT_MOL_COLUMN_NAME, InvalidMol
from scikit_mol.fingerprints import MorganFingerprintTransformer
from scikit_mol.standardizer import Standardizer

from .fixtures import smiles_list

# input SMILES -> (standardized and neutralized, standardized only)
_CASES = {
    "O=C(O)c1ccccc1": ("O=C(O)c1ccccc1", "O=C(O)c1ccccc1"),
    "O=C([O-])c1ccccc1": ("O=C(O)c1ccccc1", "O=C([O-])c1ccccc1"),
    "O=C([O-])c1ccccc1.[Na+]": ("O=C(O)c1ccccc1", "O=C([O-])c1ccccc1"),
    "O=C(O[Na])c1ccccc1": ("O=C(O)c1ccccc1", "O=C([O-])c1ccccc1"),
    "C[N+](C)C.O=C([O-])c1ccccc1": ("O=C(O)c1ccccc1", "O=C([O-])c1ccccc1"),
}


def _mols(smiles):
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def _smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in np.asarray(mols).flatten()]


@pytest.fixture
def standardizer():
    return Standardizer()


def test_standardizer_neutralizes_and_keeps_parent(standardizer):
    result = standardizer.transform(_mols(_CASES))
    assert result.shape == (len(_CASES), 1)
    assert _smiles(result) == [neutral for neutral, _ in _CASES.values()]


def test_standardizer_without_neutralize():
    result = Standardizer(neutralize=False).transform(_mols(_CASES))
    assert _smiles(result) == [charged for _, charged in _CASES.values()]


def test_standardizer_does_not_modify_input(standardizer):
    mols = _mols(_CASES)
    before = _smiles(mols)
    standardizer.transform(mols)
    assert _smiles(mols) == before


def test_standardizer_fit_returns_self(standardizer, smiles_list):
    assert standardizer.fit(_mols(smiles_list)) is standardizer
    assert standardizer.fit_transform(_mols(smiles_list)).shape == (len(smiles_list), 1)


@pytest.mark.parametrize(
    "container",
    [
        lambda mols: mols,
        lambda mols: np.array(mols),
        lambda mols: np.array(mols).reshape(-1, 1),
        lambda mols: pd.Series(mols, name="ROMol"),
        lambda mols: pd.DataFrame({"ROMol": mols}),
    ],
    ids=["list", "array_1d", "array_2d", "series", "dataframe"],
)
def test_standardizer_input_containers(standardizer, container):
    result = standardizer.transform(container(_mols(_CASES)))
    assert _smiles(result) == [neutral for neutral, _ in _CASES.values()]


def test_standardizer_get_set_params_and_clone(standardizer):
    assert standardizer.get_params() == {
        "neutralize": True,
        "n_jobs": None,
        "safe_inference_mode": False,
    }
    mols = _mols(_CASES)
    default = _smiles(standardizer.transform(mols))
    standardizer.set_params(neutralize=False)
    changed = _smiles(standardizer.transform(mols))
    assert changed != default
    assert changed == _smiles(Standardizer(neutralize=False).transform(mols))
    cloned = clone(standardizer)
    assert cloned is not standardizer
    assert cloned.get_params() == standardizer.get_params()


def test_standardizer_parallel(standardizer):
    mols = _mols(list(_CASES) * 4)
    expected = _smiles(standardizer.transform(mols))
    standardizer.set_params(n_jobs=2)
    assert _smiles(standardizer.transform(mols)) == expected


def test_standardizer_invalid_mol_raises(standardizer):
    with pytest.raises(ValueError, match="Invalid input molecule"):
        standardizer.transform(_mols(_CASES) + [None])


def test_standardizer_safe_inference_mode():
    standardizer = Standardizer(safe_inference_mode=True)
    upstream = InvalidMol("SmilesToMolTransformer", "Invalid SMILES: nonsense")
    result = standardizer.transform(_mols(_CASES) + [None, upstream]).flatten()
    assert _smiles(result[:-2]) == [neutral for neutral, _ in _CASES.values()]
    assert isinstance(result[-2], InvalidMol)
    assert "Invalid input molecule" in result[-2].error
    # An InvalidMol from an earlier step is passed on unchanged, so its origin is kept
    assert result[-1] is upstream


def test_standardizer_feature_names_out(standardizer):
    assert list(standardizer.get_feature_names_out()) == [DEFAULT_MOL_COLUMN_NAME]
    assert list(standardizer.get_feature_names_out(["mols"])) == [
        f"{DEFAULT_MOL_COLUMN_NAME}_mols"
    ]


def test_standardizer_in_pipeline():
    pipeline = make_pipeline(
        SmilesToMolTransformer(), Standardizer(), MorganFingerprintTransformer()
    )
    fps = pipeline.fit_transform(list(_CASES))
    assert fps.shape == (len(_CASES), 2048)
    # every input is a form of benzoic acid, so after standardization the fingerprints agree
    assert all(np.array_equal(fps[0], fp) for fp in fps[1:])


def test_standardizer_failure_during_standardization(monkeypatch):
    from scikit_mol import standardizer as standardizer_module

    def fail(mol):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(standardizer_module.rdMolStandardize, "Cleanup", fail)
    mols = _mols(["CCO"])
    with pytest.raises(RuntimeError, match="cleanup failed"):
        Standardizer().transform(mols)
    result = Standardizer(safe_inference_mode=True).transform(mols).flatten()
    assert isinstance(result[0], InvalidMol)
    assert "Standardization failed: cleanup failed" in result[0].error
