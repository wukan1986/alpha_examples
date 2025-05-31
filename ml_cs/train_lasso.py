import joblib
import polars as pl
from alphainspect.dtree import plot_coef_box
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression  # noqa

from ml_cs.config import MODEL_FILENAME, INPUT1_PATH, DATE, ASSET, LABEL, DATA_END, drop_columns, FWD_RET
from ml_cs.utils import load_dates, walk_forward, get_XyOther

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# %%
df = pl.read_parquet(INPUT1_PATH)
print(df.columns)
# ['date', 'asset', 'open', 'close', 'high', 'low', 'volume', 'amount', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'is_st', 'sw_l1', 'sw_l3', 'sw_l2', 'zjw', 'pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr', 'vwap', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP', '上海主板', '深圳主板', '科创板', '创业板', '北交所', 'ROCR', 'SSE50', 'CSI300', 'CSI500', 'CSI1000', 'DOJI4', 'NEXT_DOJI4', 'roe', 'volume_change', 'momentum_20', 'volume_trend', 'ma5', 'pe_trend', 'RET', 'volume_volatility', 'momentum_diff', 'volume_ratio_10', 'momentum_10', 'ma10', 'ma_bias', 'pb_roe', '短期上穿中期', '当前价格是否高于10日均线', 'score']

# 提前将不需要的列删除
# df = df.filter()
df = df.drop(drop_columns)
columns = sorted(df.columns)
print(columns)
logger.info('开始训练...')


# %%
def fit():
    # fit时，trading_dates只取了一部分
    trading_dates = load_dates(INPUT1_PATH, DATE)[:DATA_END]
    models = []

    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=5, max_train_size=None, test_size=30, gap=3):
        for start, end in (train_dt, test_dt):
            X, y, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, is_fit=True)
            break

        model = Lasso(
            alpha=0.01,
            max_iter=500,
            random_state=42,
        )
        # model = LinearRegression()
        model.fit(X, y)
        models.append(model)

    return models


def evaluate(models):
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_coef_box(models, ax=ax)
    plt.show()


# %%
models = fit()
logger.info('保存模型...')
joblib.dump(models, MODEL_FILENAME)

logger.info('加载模型...')
models = joblib.load(MODEL_FILENAME)
evaluate(models)
