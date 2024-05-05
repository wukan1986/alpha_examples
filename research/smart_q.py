"""
方正金工 聪明钱情绪因子Q

"""
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from loguru import logger

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


def func(df: pl.DataFrame) -> pl.DataFrame:
    # 与时序指标不同，这里排序用的不是时序
    df = df.sort(by="S", descending=True)
    df = df.with_columns(acc_volume_pct=pl.col("volume").cum_sum() / pl.col("volume").sum())
    # 另分一个字段，用于处理小于阈值
    df = df.with_columns(volume_1=pl.when(pl.col("acc_volume_pct") <= THRESHOLD).then(pl.col("volume")).otherwise(0))
    df = df.with_columns(
        vwap_smart=(pl.col("close") * pl.col("volume_1")).sum() / pl.col("volume_1").sum(),
        vwap_all=(pl.col("close") * pl.col("volume")).sum() / pl.col("volume").sum(),
    )
    df = df.with_columns(Q=pl.col("vwap_smart") / pl.col("vwap_all"))
    # 取值
    return df.select(pl.col("date").min().alias("start_date"),
                     pl.col("date").max().alias("end_date"),
                     pl.col('asset').first(), pl.col('Q').first())


def func_file(df: pl.DataFrame) -> pl.DataFrame:
    """处理一个月的数据"""
    df = df.rename({"code": "asset", "time": "date", "money": "amount"})
    df = df.filter(pl.col("paused") == 0)

    # 涨跌幅
    df = df.with_columns(R=pl.col("close") / pl.col("open") - 1)
    # 多加了1，防止出现除0
    df = df.with_columns(S=pl.col("R").abs() / ((pl.col("volume") + 1) ** BETA))

    # 分成时序指标
    return df.group_by("asset").map_groups(func)


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
    files = path_groupby_date(INPUT_PATH)
    # 过滤日期
    files = files["2024":]

    logger.info("start")
    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_files, list(files.groupby(files['key1'].dt.to_period('M')))))
        # polars合并
        output = pl.concat(output)
        output = output.with_columns(date=pl.col("end_date").dt.truncate("1d"))
        output.write_parquet("smart_q.parquet")
        print(output.tail())

    logger.info("done")
