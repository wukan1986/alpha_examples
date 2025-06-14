import numbers
from typing import Tuple

import pandas as pd
import polars as pl
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold


# 复制于sklearn.model_selection._split._BaseKFold
def __init__(self, n_splits, *, shuffle, random_state):
    if not isinstance(n_splits, numbers.Integral):
        raise ValueError(
            "The number of folds must be of Integral type. "
            "%s of type %s was passed." % (n_splits, type(n_splits))
        )
    n_splits = int(n_splits)

    # if n_splits <= 1:
    #     raise ValueError(
    #         "k-fold cross-validation requires at least one"
    #         " train/test split by setting n_splits=2 or more,"
    #         " got n_splits={0}.".format(n_splits)
    #     )

    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

    if not shuffle and random_state is not None:  # None is the default
        raise ValueError(
            (
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True."
            ),
        )

    self.n_splits = n_splits
    self.shuffle = shuffle
    self.random_state = random_state


def walk_forward(dates: pd.Series, n_splits: int, max_train_size: int = None, test_size: int = None, gap: int = 3):
    """前向分析

    Parameters
    ----------
    dates: pd.Series
        交易日期序列。
    n_splits: int
        分割份数
    max_train_size:int
        最大训练集期数
    test_size:int
        测试集期数
    gap:int
        训练集与测试集间隔

    Returns
    -------
    i, (train_start, train_end), (test_start, test_end)

    """
    if n_splits <= 1:
        _BaseKFold.__init__ = __init__

    tscv = TimeSeriesSplit(n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)
    for i, (train_index, test_index) in enumerate(tscv.split(dates)):
        # 时间序列划分
        train_start, train_end, test_start, test_end = [
            dates.iloc[j] for j in
            (train_index[0], train_index[-1], test_index[0], test_index[-1])
        ]
        logger.info('{}: {}/{}, {}/{}, {}/{}', i, len(train_index), len(test_index), train_start, train_end,
                    test_start, test_end)
        yield i, (train_start, train_end), (test_start, test_end)


def load_dates(path: str, date: str) -> pd.Series:
    """加载日期序列

    Parameters
    ----------
    path:str
        文件路径
    date:str
        日期字段名

    Returns
    -------
    pd.Series
        方便使用日期字符串切片

    """
    df = pl.read_parquet(path, columns=[date]).unique().sort(by=date)
    s = df.to_series(0).to_pandas()
    s.index = s.values
    return s


def get_XyOther(df: pl.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                date: str, asset: str, label: str, *fwd_ret: str, is_test: bool) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """获取X y other

    Parameters
    ----------
    df
    start
    end
    date
    asset
    label
    fwd_ret
    is_test:bool
        是否用于训练。
        fit时，X和y都不能出现null
        predict时，X不能出现null,y无限制
        但要验证predict效果时，y不能为hull

    Returns
    -------
    X: pl.DataFrame
    y: pl.DataFrame
    other: pl.DataFrame
        含date/asset/target/fwd_ret


    """

    df = df.filter(pl.col(date).is_between(start, end))
    if is_test:
        df = df.drop_nulls(subset=pl.exclude(*fwd_ret))
    else:
        df = df.drop_nulls(subset=pl.exclude(*fwd_ret, label))

    _X = df.select(date, asset, pl.exclude(date, asset, label, *fwd_ret))
    _y = df.select(date, asset, label)
    _other = df.select(date, asset, label, *fwd_ret)

    # 转换成复合索引，可正常输入到sklearn
    _X = _X.to_pandas().set_index([date, asset])
    _y = _y.to_pandas().set_index([date, asset])

    # 排序，防止不同数据源的列头顺序不同
    _X = _X[sorted([col for col in _X.columns])]

    return _X, _y, _other
