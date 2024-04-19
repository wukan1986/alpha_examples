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
import inspect

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa

import polars as pl
import polars.selectors as cs


def _code_block_3():
    # filter后计算的代码

    # TODO 打标签应当在票池中打，还是在全A中打？
    LABEL_OO_2 = cs_mad_zscore(RETURN_OO_2)
    LABEL_OO_5 = cs_mad_zscore(RETURN_OO_5)
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

    # 风控指标，不参与机器学习，但参与最后的下单过滤
    R_01 = CLOSE / ts_mean(CLOSE, 5) - 1
    R_02 = ts_mean(CLOSE, 5) / ts_mean(CLOSE, 10) - 1
    R_03 = close / 3 - 1

    A_0001 = cs_mad_zscore_resid(volume * -1, CS_SW_L1, ONE)
    A_0002 = cs_mad_zscore_resid(turnover_ratio * -1, CS_SW_L1, LOG_MC_ZS, ONE)


def code_to_string(code_block):
    source = inspect.getsource(code_block)
    raw, exprs_dict = sources_to_exprs(globals().copy(), source, safe=False)

    # 生成代码
    codes, G = ExprTool().all(exprs_dict, style='polars', template_file='template.py.j2',
                              replace=True, regroup=True, format=True,
                              date='date', asset='asset',
                              # 复制了需要使用的函数，还复制了最原始的表达式
                              extra_codes=(raw,
                                           # 传入多个列的方法
                                           r'CS_SW_L1 = pl.col(r"^sw_l1_\d+$")',
                                           # rf'CS_SW_L1 = [pl.col(i) for i in {sw_l1_columns}]',
                                           ))
    # 传入多个列的方法
    # codes = codes.replace(', CS_SW_L1', ', *CS_SW_L1')

    return codes


if __name__ == '__main__':
    # 去除停牌后的基础数据
    INPUT_PATH = r'M:\data3\T1\feature1.parquet'
    # 添加新特证，有可能因过滤问题，某些股票在票池中反复剔除和纳入
    OUTPUT_PATH = r'M:\data3\T1\feature2.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)
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
    output_file = 'research/output3.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code_to_string(_code_block_3))

    logger.info('转码完成')
    # =====================================
    # 有纳入剔除影响的过滤
    df = df.filter(~pl.col('is_st'))
    # TODO 只在中证500中计算，由于剔除和纳入的问题，收益计算发生了改变
    df = df.filter(pl.col('CSI500') > 0)
    # =====================================
    from research.output3 import main

    df = main(df)

    # 将计算结果中的inf都换成null
    df = df.with_columns(fill_nan(purify(cs.numeric())))

    logger.info('特征计算完成')
    # =====================================
    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)
