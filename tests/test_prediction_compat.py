"""Run example entry points without their import-time file/plot side effects."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_prediction_accepts_sklearn_and_lightgbm_signatures():
    class Regressor:
        def predict(self, X):
            return X[:, 0]

    class Booster:
        best_iteration = 7

        def predict(self, X, *, num_iteration):
            assert num_iteration == self.best_iteration
            return X[:, 0] * 3

    tree = ast.parse((ROOT / "ml_cs/pred.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict")
    X = np.array([[1.0], [2.0]])
    namespace = dict(pl=pl, np=np, models=[Regressor(), Booster()], df=None,
        DATA_START=0, INPUT1_PATH="unused", DATE="date", ASSET="asset", LABEL="label", FWD_RET="return",
        label_drop_nulls=False, method="regression", Method=SimpleNamespace(RealWorld="regression"),
        load_dates=lambda *args: [1, 2],
        walk_forward=lambda *args, **kwargs: [(0, [1, 2], [1, 2])],
        get_XyOther=lambda *args, **kwargs: (X, X[:, 0], pl.DataFrame({"asset": ["A", "B"]})))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pred.py", "exec"), namespace)
    assert namespace["predict"]()["y_pred"].to_list() == [2., 4.]


@pytest.mark.parametrize("path", ["data/1_prepare_data.py", "data/2_features.py"])
def test_feature_entry_points_supply_required_date_index(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "main")
    sentinel = object()
    def main(df, ge_date_idx):
        assert df is sentinel and ge_date_idx == 0
        return df
    assert eval(compile(ast.Expression(call), path, "eval"), {"main": main, "df": sentinel}) is sentinel
