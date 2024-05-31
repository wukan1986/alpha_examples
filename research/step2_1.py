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

import polars as pl
import polars.selectors as cs
from alphainspect.reports import create_1x3_sheet
from alphainspect.utils import with_factor_top_k
from expr_codegen.tool import codegen_exec
from loguru import logger
from matplotlib import pyplot as plt

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_1():
    filter = and_(~is_st,
                  SSE50 + CSI300 + CSI500 + CSI1000 > 0,
                  market_cap >= 20,
                  )


def _code_block_3():
    # filter后计算的代码

    # TODO 打标签应当在票池中打，还是在全A中打？
    LABEL_OO_02 = cs_mad_zscore(RETURN_OO_02)
    LABEL_OO_05 = cs_mad_zscore(RETURN_OO_05)
    LABEL_OO_10 = cs_mad_zscore(RETURN_OO_10)

    volatility_60 = cs_zscore(ts_std_dev(ROCR, 60))
    volatility_120 = cs_zscore(ts_std_dev(ROCR, 120))
    volume_15 = cs_zscore(ts_mean(volume, 15))
    score = -1 * (volatility_60 + volatility_120) + volume_15

    # 条件过滤
    volatility_240 = cs_rank(ts_std_dev(ROCR, 240))
    turn_7 = cs_rank(ts_delay(turnover_ratio, 7))

    return_30 = cs_rank(CLOSE / ts_delay(CLOSE, 30))
    return_90 = cs_rank(CLOSE / ts_delay(CLOSE, 90))

    filter = and_(volatility_240 <= 0.3, turn_7 >= 0.7,
                  return_30 < 0.5, return_90 > 0.5)


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
        pl.col('sw_l1').is_not_null(),  # TODO 没有行业的也过滤，这会不会有问题？
    )

    # TODO drop_first丢弃哪个字段是随机的，非常不友好，只能在行业中性化时动态修改代码
    df = df.with_columns(df.to_dummies('sw_l1', drop_first=True))
    sw_l1_columns = list(filter(lambda x: re.search(r"^sw_l1_\d+$", x), df.columns))
    print(sw_l1_columns)

    logger.info('数据准备完成')

    # =====================================
    # 有纳入剔除影响的过滤
    df = codegen_exec(df, _code_block_1).filter(pl.col('filter'))
    df = codegen_exec(df, _code_block_3)

    # 将计算结果中的inf都换成null
    df = df.with_columns(fill_nan(purify(cs.numeric())))

    df = df.filter(
        # TODO 中证500成份股可能被过滤，所以对于相关板块的过滤需要放在后面
        ~pl.col('asset').str.starts_with('68'),  # 过滤科创板
        # ~pl.col('asset').str.starts_with('30'),  # 过滤创业板
    )

    logger.info('特征计算完成')
    # =====================================
    # 推荐保存到内存盘中
    # df.write_parquet(OUTPUT_PATH)
    # logger.info('特征保存完成, {}', OUTPUT_PATH)
    # =====================================
    factor = 'score'
    fwd_ret_1 = 'RETURN_OO_02'
    forward_return = 'LABEL_OO_02'
    axvlines = ('2024-01-01',)

    # df = with_factor_quantile(df, factor, quantiles=quantiles, factor_quantile=f'_fq_{factor}')
    df = with_factor_top_k(df, factor, top_k=10, factor_quantile=f'_fq_{factor}')
    fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df, factor, forward_return, fwd_ret_1,
                                                              factor_quantile=f'_fq_{factor}',
                                                              figsize=(12, 6),
                                                              axvlines=axvlines)

    plt.show()
