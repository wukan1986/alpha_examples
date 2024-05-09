"""
20200307-开源证券-市场微观结构研究系列（5）：APM因子模型的进阶版

除了APM因子，此代码还实现了W切割
"""
import datetime
import math
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from loguru import logger
from polars_ta.wq import ts_regression_resid, ts_ir, ts_sum

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")
INPUT2_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_index_minute")
INPUT3_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_factor")

AM_T1 = datetime.time(9, 31)
AM_T2 = datetime.time(11, 30)
PM_T1 = datetime.time(13, 1)
PM_T2 = datetime.time(15, 00)

TIMES = [AM_T1, AM_T2, PM_T1, PM_T2]
TIMES_CLOSE = [AM_T2, PM_T2]


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


_ = (r"open", r"close",)
(open, close,) = (pl.col(i) for i in _)

_ = (r"R", r"R_i", r"resid", r"resid_pm", r"rsum")
(R, R_i, resid, resid_pm, rsum) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=[_DATE_])
    # ========================================
    df = df.with_columns(
        # 每一节开始时的价格前移，可用于日内的收益计算
        OPEN_shift=pl.col('OPEN').shift(),
        OPEN_i_shift=pl.col('OPEN_i').shift(),
        # 昨天的收盘价移动
        CLOSE_shift=pl.col('CLOSE').shift(),
        CLOSE_i_shift=pl.col('CLOSE_i').shift(),
    )
    df = df.with_columns(
        # 日内收益率
        RT=pl.col('CLOSE') / pl.col('OPEN_shift') - 1,
        RT_i=pl.col('CLOSE_i') / pl.col('OPEN_i_shift') / - 1,
        # 隔夜收益率
        RY=pl.col('OPEN') / pl.col('CLOSE_shift') - 1,
        RY_i=pl.col('OPEN_i') / pl.col('CLOSE_i_shift') / - 1,
    )
    return df


def get_returns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_0_ts__asset)
    return df


def func_1_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=[_DATE_])
    # ========================================
    df = df.with_columns(
        # 用于APM因子
        resid=ts_regression_resid(R, R_i, 20),
        # 用于W切割
        rsum=ts_sum(R, 20),
    )
    return df


def get_resid(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_1_ts__asset)
    return df.select(_DATE_, _ASSET_, 'date_', resid, rsum)


def func_2_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=[_DATE_])
    # ========================================
    df = df.with_columns(
        stat=ts_ir(resid - resid_pm, 20) / math.sqrt(20)
    )
    return df


def get_stat(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_2_ts__asset)
    return df


def func_files(name_group, col) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    # 股票
    dfs = []
    for path in group[col]:
        df = pl.read_parquet(path).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
        df = df.filter(pl.col(_DATE_).dt.time().is_in(TIMES))
        dfs.append(df)
    dfs = pl.concat(dfs)

    return dfs


def multi_task(f1, f2, f3):
    # 指数
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(map(lambda x: func_files(x, "path"), list(f2.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output.write_parquet("APM因子_index.parquet")

    # 股票
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(map(lambda x: func_files(x, "path"), list(f1.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output = output.with_columns(pl.col(_DATE_).dt.truncate("1d").alias('date_'))

        # 读取日线复权因子
        df0 = pl.read_parquet(f3['path'].to_list(), use_pyarrow=True).rename({"code": _ASSET_, "time": _DATE_})
        df0 = df0.with_columns(pl.col(_DATE_).cast(pl.Datetime(time_unit="ns")))

        # 分钟线合并日线复权因子
        output = output.join(df0, left_on=['date_', _ASSET_], right_on=[_DATE_, _ASSET_], how='left')

        output.write_parquet("APM因子_stock.parquet")


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    f2 = path_groupby_date(INPUT2_PATH)
    f3 = path_groupby_date(INPUT3_PATH)
    # 过滤日期
    f1 = f1["2024-01":]
    f2 = f2["2024-01":]
    f3 = f3["2024-01":]

    logger.info("start")

    # 多进程
    multi_task(f1, f2, f3)

    # 加载数据
    df1 = pl.read_parquet("APM因子_stock.parquet").filter(pl.col("paused") == 0)
    df1 = df1.with_columns([
        (pl.col(['open', 'high', 'low', 'close']) * pl.col('factor')).name.map(lambda x: x.upper()),
    ])
    df2 = pl.read_parquet("APM因子_index.parquet").filter(pl.col(_ASSET_) == "000001.XSHG")
    df2 = df2.with_columns([
        (pl.col(['open', 'high', 'low', 'close']) * 1).name.map(lambda x: x.upper()),
    ])
    df3 = df1.join(df2, on=[_DATE_], suffix="_i")
    del df1
    del df2
    logger.info("计算收益率")
    df3 = get_returns(df3)
    # 挑选指定时间点的数据
    df_RO = df3.filter(pl.col(_DATE_).dt.time() == AM_T1).select(_DATE_, _ASSET_, 'date_', R=pl.col("RY"), R_i="RY_i")
    df_RA = df3.filter(pl.col(_DATE_).dt.time() == AM_T2).select(_DATE_, _ASSET_, 'date_', R=pl.col("RT"), R_i="RT_i")
    df_RP = df3.filter(pl.col(_DATE_).dt.time() == PM_T2).select(_DATE_, _ASSET_, 'date_', R=pl.col("RT"), R_i="RT_i")
    del df3

    logger.info("计算残差")
    # 这个用时久
    df_EO = get_resid(df_RO)
    df_EA = get_resid(df_RA)
    df_EP = get_resid(df_RP)
    del df_RO
    del df_RA
    del df_RP

    logger.info("计算stat")
    df_APM_raw = df_EA.join(df_EP, on=['date_', _ASSET_], suffix='_pm')
    df_APM_new = df_EO.join(df_EP, on=['date_', _ASSET_], suffix='_pm')
    del df_EO
    del df_EA
    del df_EP
    # 计算统计量stat
    # TODO 还需要再与ret20回归一下
    df_APM_raw = get_stat(df_APM_raw).rename({'stat': 'APM_raw'})
    df_APM_new = get_stat(df_APM_new).rename({'stat': 'APM_new'})

    logger.info("W式切割")
    df_APM_raw = df_APM_raw.with_columns(AVP=pl.col("rsum") - pl.col("rsum_pm"))
    df_APM_new = df_APM_new.with_columns(OVP=pl.col("rsum") - pl.col("rsum_pm"))

    output = df_APM_raw.join(df_APM_new, on=['date_', _ASSET_]).select(_DATE_, _ASSET_, 'date_', 'APM_raw', 'APM_new', 'AVP', 'OVP')
    output.write_parquet("APM因子.parquet")
    print(output.tail())

    logger.info("done")
