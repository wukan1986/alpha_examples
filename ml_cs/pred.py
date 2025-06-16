from enum import Enum

import joblib
import numpy as np
import polars as pl
from alphainspect.reports import create_3x2_sheet
from alphainspect.utils import with_factor_quantile
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report

from ml_cs.config import DATE, ASSET, LABEL, MODEL_FILENAME, INPUT1_PATH, DATA_START, FWD_RET
from ml_cs.config import load_process_regression, load_process_binary, load_process_unbalance  # noqa
from ml_cs.utils import load_dates, get_XyOther, walk_forward

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


class Method(Enum):
    RealWorld = 1  # 真实世界，y有空值
    Binary = 2  # 使用平衡二分类数据，同时看分类报告
    Unbalance = 3  # 使用不平衡数据，同时看分类报告


method = Method.Unbalance
# %%
if method == Method.RealWorld:
    df = load_process_regression()
    label_drop_nulls = False
if method == Method.Binary:
    df = load_process_binary()
    label_drop_nulls = True
if method == Method.Unbalance:
    df = load_process_unbalance()
    label_drop_nulls = True

# %%
logger.info('加载模型...')
models = joblib.load(MODEL_FILENAME)


# %% 预测
def predict():
    trading_dates = load_dates(INPUT1_PATH, DATE)[DATA_START:]

    others = []
    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=1, max_train_size=None, test_size=None, gap=0):
        start, end = train_dt[0], test_dt[-1]

        X_test, y_test, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, label_drop_nulls=label_drop_nulls)

        y_preds = {}
        for i, model in enumerate(models):
            num_iteration = model.best_iteration if hasattr(model, 'best_iteration') else None
            pred_proba = model.predict(X_test, num_iteration=num_iteration)
            print("预测概率范围:", pred_proba.min(), "~", pred_proba.max())
            if method != Method.RealWorld:
                print("AUC分数:", roc_auc_score(y_test, pred_proba))
                print(classification_report(y_test, (pred_proba > 0.5).astype(int), zero_division=np.nan))
            y_preds[f'y_pred_{i}'] = pred_proba
        # TODO 预测值等权,可以按需进行权重分配
        result = other.with_columns(y_pred=pl.from_dict(y_preds).mean_horizontal())
        others.append(result)

    return pl.concat(others)


result = predict()
print(result)

# %%
axvlines = ()
factor = 'y_pred'
df = with_factor_quantile(result, factor, quantiles=10, factor_quantile='_fq_1')
create_3x2_sheet(df, factor, FWD_RET, factor_quantile='_fq_1', axvlines=axvlines)
plt.show()
