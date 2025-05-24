"""
20191029-东方证券- 因子选股系列研究六十：基于量价关系度量股票的买卖压力

提到了三个因子，日频因子好处理
APB_1m = log(ts_mean(VWAP, 20) / (ts_sum(VWAP * volume, 20) / ts_sum(volume, 20)))
APB_5d = ts_mean(log(ts_mean(VWAP, 5) / (ts_sum(VWAP * volume, 5) / ts_sum(volume, 5))), 20)

而APB_1d由于涉及到5分钟换成日频，需要特殊代码，由于特殊写法，不必复权

"""
import multiprocessing
import pathlib

import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.prefix.wq import log
from polars_ta.prefix.wq import ts_mean

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")

_ = (r"open", r"close", r"volume", r"amount", r"vwap")
(open, close, volume, amount, vwap) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(_DATE_)

    # 1分钟转5分钟
    df = df.group_by_dynamic(_DATE_, every="5m", closed='right').agg(
        pl.first('date_'),
        pl.first(_ASSET_),
        pl.first(_DATE_).alias("open_date"),
        pl.last(_DATE_).alias("close_date"),
        pl.first("open"), pl.max('high'), pl.min('low'),
        pl.last("close"),
        pl.sum("volume"), pl.sum("amount"),
    )
    df = df.with_columns(
        vwap=amount / volume,
    )

    # 计算成日频
    df = df.group_by("date_").agg(
        pl.first(_DATE_),
        pl.first(_ASSET_),
        apb=log(vwap.mean() / ((vwap * volume).sum() / volume.sum())),
    )

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
    df = df.sort(_ASSET_, _DATE_)

    df = df.with_columns(pl.col(_DATE_).dt.truncate("1d").alias("date_"))

    return func_file(df)


def multi_task(f1):
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output.write_parquet("买卖压力_temp.parquet")
        print(output.tail())


def _code_block_1():
    # 日内收益率
    APB_1d = ts_mean(apb, 20)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    # 过滤日期
    f1 = f1["2024-01":]

    logger.info("start")
    # 初步计算
    multi_task(f1)
    logger.info("计算")
    df = pl.read_parquet("买卖压力_temp.parquet")
    df = df.with_columns(pl.col(_DATE_).dt.truncate("1d"))

    df = codegen_exec(df, _code_block_1, over_null="partition_by")
    df.write_parquet("买卖压力.parquet")
    print(df.tail())

    logger.info("done")
