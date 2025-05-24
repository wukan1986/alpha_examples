"""
20200721-开源证券-开源量化评论（3）：A股市场中如何构造动量因子？


"""
import polars as pl
from expr_codegen import codegen_exec
from loguru import logger

from polars_ta.wq import ts_returns, ts_sum_split_by


def _code_block_1():
    WMOM1, WMOM2 = ts_sum_split_by(ts_returns(CLOSE, 1), HIGH / LOW - 1, 60, 30)


if __name__ == '__main__':
    # 去除停牌后的基础数据
    INPUT_PATH = r'M:\preprocessing\data2.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)

    df = codegen_exec(df, _code_block_1, over_null="partition_by")
    print(df.tail())
    logger.info("done")
