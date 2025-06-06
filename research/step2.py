"""
一次生成多套特征，之后在`step3.py`中将用于比较多特征之间区别

['date', 'asset', 'open', 'close', 'high', 'low', 'volume', 'amount',
'high_limit', 'low_limit', 'pre_close',
'paused', 'factor', 'is_st',
'sw_l1', 'sw_l3', 'sw_l2', 'zjw',
'pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr',
'vwap', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP',
'上海主板', '深圳主板', '科创板', '创业板', '北交所',
'ROCR', 'SSE50', 'CSI300', 'CSI500', 'CSI1000',
'DOJI4', 'NEXT_DOJI4',
'RETURN_CC_01', 'RETURN_CO_01', 'RETURN_OC_01', 'RETURN_OO_01', 'RETURN_OO_02', 'RETURN_OO_05', 'RETURN_OO_10']
"""
import os
import sys
from pathlib import Path

from alphainspect.reports import create_1x3_sheet  # noqa
from alphainspect.utils import with_factor_quantile, with_factor_top_k  # noqa
from matplotlib import pyplot as plt
from polars_ta.prefix.wq import *
from polars_ta.utils.withs import with_industry

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================

import polars as pl
import polars.selectors as cs
from expr_codegen import codegen_exec
from loguru import logger


def _code_block_1():
    # 时序类特征，一定要提前算，防止被is_st等过滤掉
    ONE = 1
    _OO_02 = OPEN[-3] / OPEN[-1]
    RETURN_OO_02 = _OO_02 ** (1 / 2) - 1


def _code_block_2():
    filter = and_(
        ~is_st,  # 过滤掉ST
        SSE50 + CSI300 + CSI500 + CSI1000 > 0,  # 过滤掉不在这4个指数中的股票
        market_cap >= 20,  # 过滤掉市值小于20亿的股票
    )


def _code_block_3():
    # TODO 打标签应当在票池中打，还是在全A中打？
    LABEL_OO_02 = cs_quantile(RETURN_OO_02, 0.01, 0.99)

    # TODO 本人尝试的指标处理方法，不知是否合适，欢迎指点
    # 对数市值。去极值
    MC_LOG = cs_quantile(log1p(market_cap), 0.01, 0.99)
    # 对数市值。标准化。供其他因子市值中性化时使用
    MC_NORM = cs_zscore(MC_LOG)
    # 对数市值。行业中性化。直接作为因子使用
    MC_NEUT = cs_resid_zscore(MC_NORM, CS_SW_L1, ONE)

    # 1 / pe_ratio和1 / pb_ratio需要处理负数吗？ cs_quantile_zscore cs_mad_zscore
    EP = cs_mad_zscore_resid_zscore(1 / pe_ratio, MC_NORM, CS_SW_L1, ONE)
    BP = cs_mad_zscore_resid_zscore(1 / pb_ratio, MC_NORM, CS_SW_L1, ONE)
    SP = cs_mad_zscore_resid_zscore(1 / ps_ratio, MC_NORM, CS_SW_L1, ONE)
    CFP = cs_mad_zscore_resid_zscore(1 / pcf_ratio, MC_NORM, CS_SW_L1, ONE)

    factor = ts_corr(turnover_ratio, ts_delay(CLOSE, 1), 20)

    POS_QTL = cs_quantile_zscore(factor)
    POS_MAD = cs_mad_zscore(factor)
    NEG_QTL = -cs_quantile_zscore(factor)
    NEG_MAD = -cs_mad_zscore(factor)


if __name__ == '__main__':
    # 去除停牌后的基础数据
    INPUT1_PATH = r'M:\preprocessing\data2.parquet'

    # 添加新特证，有可能因过滤问题，某些股票在票池中反复剔除和纳入
    OUTPUT_PATH = r'M:\preprocessing\data4.parquet'

    logger.info('准备基础数据, {}', INPUT1_PATH)
    df = pl.read_parquet(INPUT1_PATH)
    print(df.columns)
    #
    # 添加申万一级行业
    df = with_industry(df, 'sw_l1', drop_first=True, keep_col=True)
    logger.info('数据准备完成')

    # =====================================
    df = codegen_exec(df, _code_block_1, over_null="partition_by", output_file="1.py")
    df = codegen_exec(df, _code_block_2, over_null="partition_by", output_file="2.py").filter(pl.col('filter'))
    df = codegen_exec(df, _code_block_3, over_null="partition_by", output_file="3.py")

    # 将计算结果中的inf都换成null
    df = df.with_columns(purify(cs.numeric()))

    df = df.filter(
        # TODO 中证500成份股可能被过滤，所以对于相关板块的过滤需要放在后面
        ~pl.col('asset').str.starts_with('68'),  # 过滤科创板
        # ~pl.col('asset').str.starts_with('30'),  # 过滤创业板
    )

    logger.info('特征计算完成')
    # =====================================
    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)

    # TODO 不显示图表，不需要可以注释
    sys.exit()

    factor = 'factor'
    fwd_ret_1 = 'LABEL_OO_02'
    axvlines = ('2024-01-01',)
    quantiles = 9

    df = with_factor_quantile(df, factor, quantiles=quantiles, factor_quantile=f'_fq_{factor}')
    # df = with_factor_top_k(df, factor, top_k=10, factor_quantile=f'_fq_{factor}')
    fig, ic_dict, hist_dict, cum, avg, std = create_1x3_sheet(df, factor, fwd_ret_1,
                                                              factor_quantile=f'_fq_{factor}',
                                                              figsize=(12, 6),
                                                              axvlines=axvlines)

    plt.show()
