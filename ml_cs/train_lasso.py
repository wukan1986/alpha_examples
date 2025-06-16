import joblib
from alphainspect.dtree import plot_coef_box
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression  # noqa

from ml_cs.config import MODEL_FILENAME, INPUT1_PATH, DATE, ASSET, LABEL, DATA_END, FWD_RET, load_process_regression
from ml_cs.utils import load_dates, walk_forward, get_XyOther

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# %%
df = load_process_regression()
logger.info('开始训练...')


# %%
def fit():
    trading_dates = load_dates(INPUT1_PATH, DATE)[:DATA_END]
    models = []

    for i, train_dt, test_dt in walk_forward(trading_dates,
                                             n_splits=5, max_train_size=None, test_size=30, gap=3):
        for start, end in (train_dt, test_dt):
            X, y, other = get_XyOther(df, start, end, DATE, ASSET, LABEL, FWD_RET, label_drop_nulls=True)
            break

        model = Lasso(
            alpha=0.01,
            max_iter=500,
            random_state=42,
        )
        model = LinearRegression()
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
