"""
20200307-开源证券-市场微观结构研究系列（5）：APM因子模型的进阶版

除了APM因子，此代码还实现了W切割
"""
import datetime
import multiprocessing
import pathlib

import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.wq import ts_regression_resid, ts_ir, ts_sum

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")
INPUT2_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_index_minute")
INPUT3_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_factor")

AM_T1 = datetime.time(9, 31)
AM_T2 = datetime.time(11, 30)
PM_T1 = datetime.time(13, 1)
PM_T2 = datetime.time(15, 00)

TIMES = [AM_T1, AM_T2, PM_T1, PM_T2]
TIMES_CLOSE = [AM_T2, PM_T2]

_DATE_ = "date"
_ASSET_ = "asset"


def _code_block_1():
    # 日内收益率
    RT = CLOSE / OPEN[1] - 1
    RT_i = CLOSE_i / OPEN_i[1] - 1
    # 隔夜收益率
    RY = OPEN / CLOSE[1] - 1
    RY_i = OPEN_i / CLOSE_i[1] - 1


def _code_block_2():
    # 用于APM因子
    resid = ts_regression_resid(R, R_i, 20)
    # 用于W切割
    rsum = ts_sum(R, 20)


def _code_block_3():
    stat = ts_ir(resid - resid_pm, 20) / (20 ** 0.5)


def func_files(name_group) -> pl.DataFrame:
    """每月多个文件成一组"""
    name, group = name_group
    logger.info(name)

    df = pl.read_parquet(group['path'].to_list()).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df = df.filter(pl.col(_DATE_).dt.time().is_in(TIMES))

    return df


def multi_task(f1, f2, f3):
    # 指数
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f2.groupby(f1['key1'].dt.to_period('M')))))
        output = pl.concat(output)
        output.write_parquet("APM因子_index.parquet")

    # 股票
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(f1.groupby(f1['key1'].dt.to_period('M')))))
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
    df3 = codegen_exec(df3, _code_block_1)

    # 挑选指定时间点的数据
    df_RO = df3.filter(pl.col(_DATE_).dt.time() == AM_T1).select(pl.col(_DATE_).dt.truncate("1d"), _ASSET_, R=pl.col("RY"), R_i="RY_i")
    df_RA = df3.filter(pl.col(_DATE_).dt.time() == AM_T2).select(pl.col(_DATE_).dt.truncate("1d"), _ASSET_, R=pl.col("RT"), R_i="RT_i")
    df_RP = df3.filter(pl.col(_DATE_).dt.time() == PM_T2).select(pl.col(_DATE_).dt.truncate("1d"), _ASSET_, R=pl.col("RT"), R_i="RT_i")
    del df3

    logger.info("计算残差")
    # 这个用时久
    df_EO = codegen_exec(df_RO, _code_block_2)
    df_EA = codegen_exec(df_RA, _code_block_2)
    df_EP = codegen_exec(df_RP, _code_block_2)

    del df_RO
    del df_RA
    del df_RP

    logger.info("计算stat")
    df_APM_raw = df_EA.join(df_EP, on=[_DATE_, _ASSET_], suffix='_pm')
    df_APM_new = df_EO.join(df_EP, on=[_DATE_, _ASSET_], suffix='_pm')
    del df_EO
    del df_EA
    del df_EP
    # 计算统计量stat
    # TODO 还需要再与ret20回归一下
    df_APM_raw = codegen_exec(df_APM_raw, _code_block_3).rename({'stat': 'APM_raw'})
    df_APM_new = codegen_exec(df_APM_new, _code_block_3).rename({'stat': 'APM_new'})

    logger.info("切割")
    df_APM_raw = df_APM_raw.with_columns(AVP=pl.col("rsum") - pl.col("rsum_pm"))
    df_APM_new = df_APM_new.with_columns(OVP=pl.col("rsum") - pl.col("rsum_pm"))

    output = df_APM_raw.join(df_APM_new, on=[_DATE_, _ASSET_]).select(_DATE_, _ASSET_, 'APM_raw', 'APM_new', 'AVP', 'OVP')
    output.write_parquet("APM因子.parquet")
    print(output.tail())

    logger.info("done")
