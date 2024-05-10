"""
20200209-开源证券-市场微观结构研究系列（3）：聪明钱因子模型的2.0版本
"""
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from loguru import logger
from polars_ta.wq import abs_

INPUT_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")
# 每月取最近10个交易日
DAYS = 10
BETAS = [-0.5, -0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25, 0.33, 0.5, 0.7]
BETAS_LOG = BETAS + ['log']
# 阈值
THRESHOLD = 0.2


def path_groupby_date(input_path: pathlib.Path) -> pd.DataFrame:
    """将文件名中的时间提取出来"""
    files = list(input_path.glob(f'*'))

    # 提取文件名中的时间
    df = pd.DataFrame([f.name.split('.')[0].split("__") for f in files], columns=['start', 'end'])
    df['path'] = files
    df['key1'] = pd.to_datetime(df['start'])
    df['key2'] = df['key1']
    df.index = df['key1'].copy()
    df.index.name = 'date'  # 防止无法groupby
    return df


_ = (r"open", r"close", r"volume", r"volume_1")
(open, close, volume, volume_1) = (pl.col(i) for i in _)

_ = (r"R",)
(R,) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func(df: pl.DataFrame) -> pl.DataFrame:
    # 与时序指标不同，这里排序用的不是时序
    for beta in BETAS_LOG:
        df = df.sort(by=f"S_{beta}", descending=True)
        df = df.with_columns(acc_volume_pct=volume.cum_sum() / volume.sum())

        # 另分一个字段，用于处理小于阈值
        df = df.with_columns(volume_1=pl.when(pl.col("acc_volume_pct") <= THRESHOLD).then(volume).otherwise(0))
        df = df.with_columns(
            vwap_smart=(close * volume_1).sum() / volume_1.sum(),
            vwap_all=(close * volume).sum() / volume.sum(),
        )
        df = df.with_columns(
            (pl.col("vwap_smart") / pl.col("vwap_all")).alias(f"Q_{beta}")
        )
    # 取值
    return df.select(pl.col(_DATE_).min().alias("start_date"),
                     pl.col(_DATE_).max().alias("end_date"),
                     pl.col(_ASSET_).first(),
                     pl.col([f"Q_{beta}" for beta in BETAS_LOG]).first())


def func_file(df: pl.DataFrame) -> pl.DataFrame:
    """处理一个月的数据"""

    # 涨跌幅
    df = df.with_columns(R=close / open - 1)
    # 多加了1，防止出现除0
    df = df.with_columns(
        [(abs_(R) / (volume + 1) ** beta).alias(f"S_{beta}") for beta in BETAS]
    )
    # 另一种方法
    df = df.with_columns((abs_(R) / volume.log1p()).alias("S_log"))

    # 分成时序指标
    return df.group_by(_ASSET_).map_groups(func)


def func_files(name_group) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    group = group.tail(DAYS)
    df = pl.read_parquet(group['path'].to_list()).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df = df.filter(pl.col("paused") == 0)

    return func_file(df)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT_PATH)
    # 过滤日期
    f1 = f1["2024":]

    logger.info("start")
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
        # polars合并
        output = pl.concat(output)
        output = output.with_columns(date=pl.col("end_date").dt.truncate("1d"))
        output.write_parquet("聪明钱因子.parquet")
        print(output.tail())

    logger.info("done")
