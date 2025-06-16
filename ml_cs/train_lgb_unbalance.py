from copy import deepcopy
from enum import Enum

import joblib
import lightgbm as lgb
from alphainspect.dtree import plot_metric_errorbar, plot_importance_box
from imblearn.over_sampling import RandomOverSampler  # noqa
from imblearn.under_sampling import RandomUnderSampler  # noqa
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.utils import compute_sample_weight  # noqa

from ml_cs.config import MODEL_FILENAME, INPUT1_PATH, DATE, ASSET, LABEL, DATA_END, FWD_RET, categorical_feature
from ml_cs.config import load_process_regression, load_process_binary, load_process_unbalance  # noqa
from ml_cs.utils import load_dates, walk_forward, get_XyOther

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


class Method(Enum):
    OverSampling = 1
    ClassWeight = 2
    IsUnbalance = 3


# TODO 问：为何任何处理方式预测结果的IC值都是负数？而源特征的IC都是正数
# 是分布改变了？
method = Method.OverSampling

# %%
params = {
    'max_depth': -1,
    'num_leaves': 63,
    'min_data_in_leaf': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': 1,  # -1不显示
    'device_type': 'cpu',
    'seed': 42,
}
if method == Method.IsUnbalance:
    # 通过参数调整权重
    params.update({'objective': 'binary', 'metric': {'auc'}, "is_unbalance": True})
else:
    # 转化成平衡问题
    params.update({'objective': 'binary', 'metric': {'binary_logloss'}})
# %%
df = load_process_unbalance()
logger.info('开始训练...')


# %%
def fit():
    trading_dates = load_dates(INPUT1_PATH, DATE)[:DATA_END]

    models = []
    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=1, max_train_size=None, test_size=60, gap=3):
        ds = []
        for start, end in (train_dt, test_dt):
            X, y, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, label_drop_nulls=True)

            if method == Method.OverSampling:
                print(y[LABEL].value_counts())
                sampler = RandomOverSampler(random_state=42)
                X, y = sampler.fit_resample(X, y)

            sample_w = None
            if method == Method.ClassWeight:
                class_w = {0: 1, 1: 5}  # TODO 根据实际情况调整
                sample_w = compute_sample_weight(class_weight=class_w, y=y)

            if len(ds) == 0:
                ds.append(lgb.Dataset(X, label=y, categorical_feature=categorical_feature, weight=sample_w))
            else:
                ds.append(lgb.Dataset(X, label=y, categorical_feature=categorical_feature, weight=sample_w, reference=ds[0]))

        evals_result = {}
        model = lgb.train(
            params,
            train_set=ds[0],
            num_boost_round=500,
            valid_sets=ds,
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(50, first_metric_only=False, verbose=True),
                lgb.record_evaluation(evals_result)
            ],
        )
        # 这里非常重要，否则无法画损失图
        model.evals_result_ = deepcopy(evals_result)
        models.append(model)
    return models


# %% 模型评估
def evaluate(models):
    for metric in params['metric']:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        plot_metric_errorbar(models, metric=metric, ax=ax)
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_importance_box(models, ax=ax)
    plt.show()


# %%
models = fit()
logger.info('保存模型...')
joblib.dump(models, MODEL_FILENAME)

logger.info('加载模型...')
models = joblib.load(MODEL_FILENAME)
evaluate(models)
