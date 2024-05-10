"""
20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？


"""
import polars as pl
from loguru import logger
from polars_ta.utils.numba_ import roll_split_i2_o2

from polars_ta.wq import ts_returns

_ = (r"OPEN", r"HIGH", r"LOW", r"CLOSE",)
(OPEN, HIGH, LOW, CLOSE) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(_DATE_)
    # ========================================
    df = df.with_columns(
        WMOM=roll_split_i2_o2(ts_returns(CLOSE, 1), HIGH / LOW - 1, 60, 30).struct.field('split_a'),
    )
    return df


def get_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_0_ts__asset)
    return df


if __name__ == '__main__':
    # 去除停牌后的基础数据
    INPUT_PATH = r'M:\data3\T1\feature1.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)

    df = df.filter(pl.col("paused") == 0)
    df = df.sort(by=[_DATE_, _ASSET_])
    df = get_0_ts__asset(df)
    print(df.tail())
    logger.info("done")
