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
# 成交量**0.5
BETA = 0.5
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
    df = df.sort(by="S", descending=True)
    df = df.with_columns(acc_volume_pct=volume.cum_sum() / volume.sum())
    # 另分一个字段，用于处理小于阈值
    df = df.with_columns(volume_1=pl.when(pl.col("acc_volume_pct") <= THRESHOLD).then(volume).otherwise(0))
    df = df.with_columns(
        vwap_smart=(close * volume_1).sum() / volume_1.sum(),
        vwap_all=(close * volume).sum() / volume.sum(),
    )
    df = df.with_columns(Q=pl.col("vwap_smart") / pl.col("vwap_all"))
    # 取值
    return df.select(pl.col(_DATE_).min().alias("start_date"),
                     pl.col(_DATE_).max().alias("end_date"),
                     pl.col(_ASSET_).first(), pl.col('Q').first())


def func_file(df: pl.DataFrame) -> pl.DataFrame:
    """处理一个月的数据"""
    df = df.rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df = df.filter(pl.col("paused") == 0)

    # 涨跌幅
    df = df.with_columns(R=close / open - 1)
    # 多加了1，防止出现除0
    df = df.with_columns(S=abs_(R) / ((volume + 1) ** BETA))
    # 另一种方法
    # df = df.with_columns(S=abs_(R) / volume.log1p())

    # 分成时序指标
    return df.group_by(_ASSET_).map_groups(func)


def func_files(name_group) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    dfs = []
    # 取后10个交易日
    group = group.tail(DAYS)
    for path in group['path']:
        dfs.append(pl.read_parquet(path))

    dfs = pl.concat(dfs)
    return func_file(dfs)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT_PATH)
    # 过滤日期
    f1 = f1["2023":]

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
