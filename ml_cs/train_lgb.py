from copy import deepcopy

import joblib
import lightgbm as lgb
from alphainspect.dtree import plot_metric_errorbar, plot_importance_box
from loguru import logger
from matplotlib import pyplot as plt

from ml_cs.config import MODEL_FILENAME, INPUT1_PATH, DATE, ASSET, LABEL, DATA_END, FWD_RET, load_process, categorical_feature
from ml_cs.utils import load_dates, walk_forward, get_XyOther

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# %%
params = {
    'boosting_type': 'gbdt',
    'objective': 'mse',  # 损失函数
    # 'metric': 'None',  # 评估函数，这里用feval来替代

    'max_depth': 8,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'min_data_in_leaf': 50,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'max_bin': 127,
    'verbose': -1,  # 不显示
    'device_type': 'cpu',
    'seed': 42,
    'force_col_wise': True,
}
# %%
df = load_process()
logger.info('开始训练...')


# %%
def fit():
    # fit时，trading_dates只取了一部分
    trading_dates = load_dates(INPUT1_PATH, DATE)[:DATA_END]

    models = []
    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=3, max_train_size=None, test_size=60, gap=3):
        ds = []
        for start, end in (train_dt, test_dt):
            X, y, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, is_fit=True)
            ds.append(lgb.Dataset(X, label=y, categorical_feature=categorical_feature))

        evals_result = {}  # to record eval results for plotting
        model = lgb.train(
            params,
            ds[0],
            num_boost_round=500,
            valid_sets=ds,
            valid_names=['train', 'valid'],
            feval=None,  # 与早停相配合
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(50),  # 与fevel配合使用
                lgb.record_evaluation(evals_result)
            ],
        )
        # 这里非常重要，否则无法画损失图
        model.evals_result_ = deepcopy(evals_result)
        models.append(model)
    return models


# %% 模型评估
def evaluate(models):
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_metric_errorbar(models, metric='l2', ax=ax)
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
