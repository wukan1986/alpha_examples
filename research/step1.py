import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
from datetime import datetime
from loguru import logger

import polars as pl
import polars.selectors as cs
from expr_codegen.tool import codegen_exec

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_1():
    # 不能跳过停牌的相关信息。如成份股相关处理

    # 注意：收益没有减1，停牌时值为1。也没有平移
    ROCR = CLOSE / ts_delay(CLOSE, 1)

    # 不少成份股数据源每月底更新，而不是每天更新，所以需要用以下方法推算
    # 注意1：在成份股调整月，如果缺少调整日的权重信息当月后一段的数据不准确
    # 注意2：不在成份股的权重要为0，否则影响之后计算，所以停牌也得保留
    # SSE50 = cs_scale(ts_zip_prod(cs_fill_zero(sz50), ROCR), 100)
    # CSI300 = cs_scale(ts_zip_prod(cs_fill_zero(hs300), ROCR), 100)
    CSI500 = cs_scale(ts_zip_prod(cs_fill_zero(zz500), ROCR), 100)
    # CSI1000 = cs_scale(ts_zip_prod(cs_fill_zero(zz1000), ROCR), 100)


def _code_block_2():
    # 跳过停牌的相关指标

    # 这里用未复权的价格更合适
    DOJI = four_price_doji(open, high, low, close)
    # 明日停牌
    NEXT_DOJI = ts_delay(DOJI, -1)

    # # 远期收益率,由于平移过,含未来数据，只能用于打标签，不能用于训练
    # RETURN_OC_1 = ts_delay(CLOSE, -1) / ts_delay(OPEN, -1) - 1
    # RETURN_CC_1 = ts_delay(CLOSE, -1) / CLOSE - 1
    # RETURN_CO_1 = ts_delay(OPEN, -1) / CLOSE - 1
    RETURN_OO_1 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1) - 1
    RETURN_OO_2 = ts_delay(OPEN, -3) / ts_delay(OPEN, -1) - 1
    RETURN_OO_5 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1) - 1
    # RETURN_OO_10 = ts_delay(OPEN, -11) / ts_delay(OPEN, -1) - 1


# =======================================
# %% 生成因子


if __name__ == '__main__':
    # 由于读写多，推荐放到内存盘，加快速度
    INPUT_PATH = r'M:\data3\T1\data.parquet'
    # 去除停牌后的基础数据
    OUTPUT_PATH = r'M:\data3\T1\feature1.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)
    df = df.rename({'time': 'date', 'code': 'asset', 'money': 'amount'})
    print(df.columns)

    df = df.with_columns([
        # 添加常数列，回归等场景用得上
        pl.lit(1, dtype=pl.Float32).alias('ONE'),
        # 成交均价，未复权
        (pl.col('amount') / pl.col('volume')).alias('vwap'),
        # 成交额与成交量对数处理
        pl.col('amount').log1p().alias('LOG_AMOUNT'),
        pl.col('volume').log1p().alias('LOG_VOLUME'),
        pl.col('market_cap').log1p().alias('LOG_MC'),
        pl.col('circulating_market_cap').log1p().alias('LOG_FC'),
        # 行业处理，由浮点改成整数
        pl.col('sw_l1', 'sw_l2', 'sw_l3').cast(pl.UInt32),
    ])

    # 后复权
    df = df.with_columns([
        (pl.col(['open', 'high', 'low', 'close', 'vwap']) * pl.col('factor')).name.map(lambda x: x.upper()),
    ]).fill_nan(None)  # nan填充成null

    logger.info('数据准备完成')
    # =====================================

    df = codegen_exec(df, _code_block_1)

    # 检查成份股权重是否正确
    # df.group_by('date').agg(pl.sum('zz500'), pl.sum('CSI500')).sort('date').to_pandas()
    # =====================================
    df = df.filter(
        pl.col('date') >= datetime(2018, 1, 1),  # 过滤要测试用的数据时间范围
        pl.col('paused') == 0,  # 过滤停牌，之后才能算收益与打标签
    )

    df = codegen_exec(df, _code_block_2)

    # 计算出来的结果需要进行部分修复，防止之后计算时出错
    df = df.with_columns(pl.col('NEXT_DOJI').fill_null(False))

    # 将计算结果中的inf都换成null
    df = df.with_columns(fill_nan(purify(cs.numeric())))

    logger.info('特征计算完成')

    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)
