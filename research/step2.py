"""
一次生成多套特征，之后在`step3.py`中将用于比较多特征之间区别
"""
import os
import sys

from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import re

from expr_codegen.tool import codegen_exec
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa

import polars as pl
import polars.selectors as cs


def _code_block_3():
    # filter后计算的代码

    # TODO 打标签应当在票池中打，还是在全A中打？
    LABEL_OO_02 = cs_mad_zscore(RETURN_OO_02)
    LABEL_OO_05 = cs_mad_zscore(RETURN_OO_05)
    # LABEL_OO_10 = cs_mad_zscore(RETURN_OO_10)

    # TODO 本人尝试的指标处理方法，不知是否合适，欢迎指点
    # 对数市值。去极值，标准化
    LOG_MC_ZS = cs_mad_zscore(LOG_MC)
    # # 对数市值。行业中性化
    # LOG_MC_NEUT = cs_mad_zscore_resid(LOG_MC_ZS, CS_SW_L1, ONE)
    # # 非线性市值，中市值因子
    # LOG_MC_NL = cs_mad_zscore(cs_resid(LOG_MC ** 3, LOG_MC, ONE))
    # # 为何2次方看起来与3次方效果一样？
    # LOG_MC_NL = cs_mad_zscore(cs_resid(LOG_MC ** 2, LOG_MC, ONE))

    # TODO 风控指标，不参与因子计算前的过滤，不参与机器学习，但参与最后的下单过滤
    R_01 = CLOSE / ts_mean(CLOSE, 5)  # 市价在5日均线上方
    R_02 = ts_mean(CLOSE, 5) / ts_mean(CLOSE, 20)  # 5日均线在20日均线上方
    R_03 = close  # 低于3元的股价不考虑
    R_04 = market_cap  # 市值20亿下算微盘股

    # 原表达式
    # _1 = log(ts_mean(VWAP, 20) / (ts_sum(VWAP * volume, 20) / ts_sum(volume, 20)))
    _1 = ts_mean(log(ts_mean(VWAP, 5) / (ts_sum(VWAP * volume, 5) / ts_sum(volume, 5))), 20)

    # 去极值、标准化、中性化
    F_11 = cs_mad_zscore(_1)
    F_12 = cs_mad_zscore_resid(_1, LOG_MC_ZS, ONE)
    F_13 = cs_mad_zscore_resid(_1, CS_SW_L1, ONE)
    F_14 = cs_mad_zscore_resid(_1, CS_SW_L1, LOG_MC_ZS, ONE)

    # 中性化
    # F_11 = _1
    # F_12 = cs_resid(_1, LOG_MC_ZS, ONE)
    # F_13 = cs_resid(_1, CS_SW_L1, ONE)
    # F_14 = cs_resid(_1, CS_SW_L1, LOG_MC_ZS, ONE)

    _00 = F_11
    # 非线性处理，rank平移后平方
    # F_010 = cs_rank2(F_00, 0.10) * -1
    # F_015 = cs_rank2(F_00, 0.15) * -1
    # F_020 = cs_rank2(F_00, 0.20) * -1
    # F_025 = cs_rank2(F_00, 0.25) * -1
    # F_030 = cs_rank2(F_00, 0.30) * -1
    # F_035 = cs_rank2(F_00, 0.35) * -1
    # F_040 = cs_rank2(F_00, 0.40) * -1
    # F_045 = cs_rank2(F_00, 0.45) * -1
    F_050 = cs_rank2(_00, 0.50) * -1
    F_055 = cs_rank2(_00, 0.55) * -1
    F_060 = cs_rank2(_00, 0.60) * -1
    F_065 = cs_rank2(_00, 0.65) * -1
    F_070 = cs_rank2(_00, 0.70) * -1
    #
    # F_065 = cs_rank2(cs_mad_zscore(_1), 0.65) * -1


if __name__ == '__main__':
    # 去除停牌后的基础数据
    INPUT1_PATH = r'M:\data3\T1\feature1.parquet'

    # 添加新特证，有可能因过滤问题，某些股票在票池中反复剔除和纳入
    OUTPUT_PATH = r'M:\data3\T1\feature2.parquet'

    logger.info('准备基础数据, {}', INPUT1_PATH)
    df = pl.read_parquet(INPUT1_PATH)

    # 演示从其它地方合并数据
    # INPUT2_PATH = r'D:\GitHub\alpha_examples\reports\买卖压力TWAP.parquet'
    # logger.info('准备扩展数据, {}', INPUT2_PATH)
    # df2 = pl.read_parquet(INPUT2_PATH, columns=['date', 'asset', 'twap', 'ARPP'])
    # df2 = df2.with_columns(pl.col('date').cast(pl.Datetime(time_unit='us')))
    # df = df.join(df2, on=['date', 'asset'], how='left')
    # df.filter(pl.col('date') >= datetime(2018, 1, 1))

    print(df.columns)
    # 没有纳入剔除影响的过滤可以提前做
    df = df.filter(
        # TODO 中证500成份股可能被过滤，这里要注意
        # ~pl.col('asset').str.starts_with('68'),  # 过滤科创板
        # ~pl.col('asset').str.starts_with('30'),  # 过滤创业板
        pl.col('sw_l1').is_not_null(),  # TODO 没有行业的也过滤，这会不会有问题？
    )

    # TODO drop_first丢弃哪个字段是随机的，非常不友好，只能在行业中性化时动态修改代码
    df = df.with_columns(df.to_dummies('sw_l1', drop_first=True))
    sw_l1_columns = list(filter(lambda x: re.search(r"^sw_l1_\d+$", x), df.columns))
    print(sw_l1_columns)

    logger.info('数据准备完成')

    # =====================================
    # 有纳入剔除影响的过滤
    df = df.filter(~pl.col('is_st'))
    # 市值大于20亿才考虑，本来应当是20e8，但这里单亿是亿
    # df = df.filter(pl.col('market_cap') > 20)
    # TODO 只在中证500中计算，由于剔除和纳入的问题，收益计算发生了改变
    # df = df.filter(pl.col('CSI500') > 0)
    # =====================================
    df = codegen_exec(df, _code_block_3)

    # 将计算结果中的inf都换成null
    df = df.with_columns(fill_nan(purify(cs.numeric())))

    logger.info('特征计算完成')
    # =====================================
    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)
