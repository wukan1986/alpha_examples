"""
20171122 光大证券 基于K线最短路径构造的非流动性因子——多因子系列报告之七

有两种定义
"""
import multiprocessing
import pathlib

import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.wq import abs_, ts_mean

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")

_ = (r"open", r"high", r"low", r"close", r"amount", r"ShortCut")
(open, high, low, close, amount, ShortCut) = (pl.col(i) for i in _)

_ = (r"R",)
(R,) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    # ========================================
    df = df.with_columns(
        ShortCut=2 * (high - low) - abs_(open - close)
    )
    df = df.select(
        pl.col(_DATE_).first(),
        pl.col(_ASSET_).first(),
        # 原始定义
        ILLIQ_1=ShortCut.sum() / (amount + 1).sum(),
        # 变形定义
        ILLIQ_2=(ShortCut / (amount + 1)).sum(),
    )
    return df


def get_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_0_ts__asset)
    return df


def func_file(idx_row):
    idx, row = idx_row
    logger.info(idx)
    df1 = pl.read_parquet(row['path']).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df1 = df1.filter(pl.col("paused") == 0)
    df1 = get_0_ts__asset(df1)
    return df1


def _code_block_1():
    ILLIQ_1_MA_5 = ts_mean(ILLIQ_1, 5)
    ILLIQ_2_MA_5 = ts_mean(ILLIQ_2, 5)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    # 过滤日期
    f1 = f1["2024-04":]

    logger.info("start")

    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(map(func_file, list(f1.iterrows())))
        # polars合并
        output = pl.concat(output)
        output = output.with_columns(pl.col(_DATE_).dt.truncate("1d"))
        output = codegen_exec(output, _code_block_1, over_null="partition_by")
        output.write_parquet("K线非流动性因子.parquet")
        print(output.tail())

    logger.info("done")
