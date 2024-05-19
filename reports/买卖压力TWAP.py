"""
20200421-东方证券-《因子选股系列之六十六》：基于时间尺度度量的日内买卖压力

avg_price = (OPEN + HIGH + LOW + CLOSE) / 4
twap = ts_mean(avg_price, 20)
ARPP = (twap - ts_min(LOW, 20)) / (ts_max(HIGH, 20) - ts_min(LOW, 20))
ARPP_20d_20d = ts_mean(ARPP, 20)


"""
import multiprocessing
import pathlib

import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.prefix.wq import ts_mean

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")

_ = (r"open", r"high", r"low", r"close", r"volume", r"amount", r"twap")
(open, high, low, close, volume, amount, twap) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(_DATE_)

    # 计算成日频
    df = df.group_by("date_").agg(
        pl.first(_DATE_),
        pl.first(_ASSET_),
        twap=((open + high + low + close) / 4).mean(),
        high=high.max(),
        low=low.min(),
    )

    df = df.select(_DATE_, _ASSET_, "twap", "high", "low", ARPP=(twap - low) / (high - low + 1e-8))

    # 取值
    return df


def func_file(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(_ASSET_).map_groups(func)


def func_files(name_group) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    df = pl.read_parquet(group['path'].to_list()).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df = df.filter(pl.col("paused") == 0)

    df = df.with_columns(pl.col(_DATE_).dt.truncate("1d").alias("date_"))

    return func_file(df)


def multi_task(f1):
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output.write_parquet("买卖压力TWAP_temp.parquet")
        print(output.tail())


def _code_block_1():
    # 日内收益率
    ARPP_1d_20d = ts_mean(ARPP, 20)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    # 过滤日期
    f1 = f1["2023-01":]

    logger.info("start")
    # 初步计算
    multi_task(f1)
    logger.info("计算")
    df = pl.read_parquet("买卖压力TWAP_temp.parquet")
    df = df.with_columns(pl.col(_DATE_).dt.truncate("1d"))
    df = codegen_exec(df, _code_block_1)
    df.write_parquet("买卖压力TWAP.parquet")
    print(df.tail())

    logger.info("done")
