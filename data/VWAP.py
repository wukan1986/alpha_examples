"""
回测时，直接按开盘价或收盘价来计算收益都有非常大的差异
可以使用全天的VWAP来计算收益，但误差非常大

使用分钟VWAP会更合适。比如开盘5分钟内入场，收盘5分钟内出场

"""
import multiprocessing
import pathlib

import polars as pl
from loguru import logger

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")

_ = (r"open", r"close", r"volume", r"amount", r"vwap")
(open, close, volume, amount, vwap) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(_DATE_)

    # 1分钟转5分钟，请按自己的需求进行调整
    df = df.group_by_dynamic(_DATE_, every="5m", closed='right').agg(
        # pl.first('date_'),
        pl.first(_ASSET_),
        # pl.first(_DATE_).alias("open_date"),
        # pl.last(_DATE_).alias("close_date"),
        # pl.first("open"), pl.max('high'), pl.min('low'),
        # pl.last("close"),

        # 5分钟VWAP，未复权
        vwap=pl.sum("amount") / pl.sum("volume"),
    )

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

    return func_file(df)


def multi_task(f1):
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)

        # 挑选时间进行保存
        # output = output.filter(pl.col(_DATE_).dt.time().is_in([
        #     datetime.time(9, 30),
        #     datetime.time(14, 55)
        # ]))

        output.write_parquet("vwap_5m.parquet")
        print(output.tail())


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    # 过滤日期
    f1 = f1["2023-01":]

    logger.info("start")
    # 初步计算
    multi_task(f1)

    logger.info("done")
