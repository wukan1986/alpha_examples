"""
2016.04.06 方正证券 异动罗盘：寻一只特立独行的票


步骤（2）中“纵向标准化”的具体做法为：将当
日相关系数 rho_(s,t)减去相关系数过去 40 个交易日的均值，再
除以 40 个交易日的标准差。步骤（3）中阈值λ取 -3。筛选
方案的直观意义是，选出相关系数处于历史平均水平 3 倍标准
差以下的交易日

（1）以每日“特立独行”异动样本为原始股票池；
（2）选取其中属于“逆势涨”的股票；
（3）剔除其中近期涨幅过高的股票。

“逆势涨”以“当日个股超额收益大于 0”为近似判断；
“近期涨幅过高”以“过去 10个交易日累积超额收益大于20%”为判断标准

其它
 剔除当日全天涨停或全天跌停的股票
 剔除次日成交金额小于 200 万元的股票

"""
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from expr_codegen.tool import codegen_exec
from loguru import logger
from polars_ta.wq import ts_returns

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")
INPUT2_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_index_minute")

_ = (r"close",)
(close,) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def _code_block_1():
    # 每一节开始时的价格前移，可用于日内的收益计算
    R = ts_returns(close, 1)


def func_1_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(
        pl.col(_DATE_).first(),
        pl.col(_ASSET_).first(),
        corr=pl.corr("R", "R_index")
    )
    return df


def get_1_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_1_ts__asset)
    return df


def func_2files(idx_row):
    idx, row = idx_row
    logger.info(idx)
    df1 = pl.read_parquet(row['path_x']).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df2 = pl.read_parquet(row['path_y']).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})

    df1 = codegen_exec(df1.filter(pl.col("paused") == 0), _code_block_1)
    df2 = codegen_exec(df2.filter(pl.col(_ASSET_) == "000001.XSHG"), _code_block_1)

    dd = df1.join(df2, on=_DATE_, suffix='_index')
    d1 = get_1_ts__asset(dd)
    return d1


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    f2 = path_groupby_date(INPUT2_PATH)
    ff = pd.merge(f1, f2, left_index=True, right_index=True, how='left')
    # 过滤日期
    ff = ff["2024-01":]

    logger.info("start")

    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_2files, list(ff.iterrows())))
        # polars合并
        output = pl.concat(output)
        output = output.with_columns(pl.col(_DATE_).dt.truncate("1d"))
        output.write_parquet("特立独行.parquet")
        print(output.tail())

    logger.info("done")
