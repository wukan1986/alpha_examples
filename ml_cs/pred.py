import joblib
import polars as pl
from alphainspect.reports import create_3x2_sheet
from alphainspect.utils import with_factor_quantile
from loguru import logger
from matplotlib import pyplot as plt

from ml_cs.config import DATE, ASSET, LABEL, MODEL_FILENAME, INPUT1_PATH, DATA_START, FWD_RET, load_process
from ml_cs.utils import load_dates, get_XyOther, walk_forward

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# %%
df = load_process()
logger.info('开始预测...')

# %%
logger.info('加载模型...')
models = joblib.load(MODEL_FILENAME)


# %% 预测
def predict():
    trading_dates = load_dates(INPUT1_PATH, DATE)[DATA_START:]

    others = []
    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=3, max_train_size=None, test_size=None, gap=0):
        start, end = train_dt[0], test_dt[-1]
        X, y, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, is_fit=False)

        y_preds = {}
        for i, model in enumerate(models):
            # print(f'{i}: {model.__class__.__name__}')
            if hasattr(model, 'best_iteration'):
                y_preds[f'y_pred_{i}'] = model.predict(X, num_iteration=model.best_iteration)
            else:
                y_preds[f'y_pred_{i}'] = model.predict(X)
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
