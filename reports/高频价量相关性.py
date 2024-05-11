"""
20200223_东吴证券_“技术分析拥抱选股因子”系列研究（一）：高频价量相关性，意想不到的选股因子
"""
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.prefix.talib import ts_LINEARREG_SLOPE
from polars_ta.wq import ts_mean, ts_std_dev, cs_zscore

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")


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


_ = (r"open", r"close", r"volume", r"corr")
(open, close, volume, corr) = (pl.col(i) for i in _)

_ = (r"R",)
(R,) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func(df: pl.DataFrame) -> pl.DataFrame:
    df1 = df.select(
        pl.col(_DATE_).first(),
        pl.col(_ASSET_).first(),

        corr=pl.corr(close, volume),
    )
    # 取值
    return df1


def func_file(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(_DATE_, _ASSET_).map_groups(func)


def func_files(name_group) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    df = pl.read_parquet(group['path'].to_list()).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df = df.filter(pl.col("paused") == 0)

    df = df.with_columns(pl.col(_DATE_).dt.truncate("1d"))

    return func_file(df)


def multi_task(f1):
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output.write_parquet("高频价量相关性_temp.parquet")
        print(output.tail())


def _code_block_1():
    avg = ts_mean(corr, 20)
    std = ts_std_dev(corr, 20)
    beta = ts_LINEARREG_SLOPE(corr, 20)

    # TODO 各种中性化都没有做

    pv_corr = cs_zscore(avg) + cs_zscore(std)
    CPV = cs_zscore(pv_corr) + cs_zscore(beta)


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    f2 = path_groupby_date(INPUT1_PATH)
    # 过滤日期
    f1 = f1["2024-01":]

    logger.info("start")
    # 初步计算
    # multi_task(f1)
    logger.info("计算")
    df = pl.read_parquet("高频价量相关性_temp.parquet")
    df = df.fill_nan(None)
    df = codegen_exec(globals().copy(), _code_block_1, df,
                      extra_codes="from polars_ta.prefix.talib import ts_LINEARREG_SLOPE")

    # 这里要做一次市值中性化
    print(df.tail())

    logger.info("done")
