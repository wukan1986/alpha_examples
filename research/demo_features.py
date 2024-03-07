"""
一次生成多套特征，之后在`demo_html.py`中将用于比较多特征之间区别
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
import inspect
from datetime import datetime

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa

import polars as pl
import polars.selectors as cs


def _code_block_1():
    # 不能提前filter的代码，提前filter会导致数据错误

    # 这里用未复权的价格更合适
    DOJI = four_price_doji(open, high, low, close)
    NEXT_DOJI = ts_delay(DOJI, -1)

    # 远期收益率
    RETURN_OO_1 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1) - 1
    RETURN_OO_2 = ts_delay(OPEN, -3) / ts_delay(OPEN, -1) - 1
    RETURN_OO_5 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1) - 1
    RETURN_OO_10 = ts_delay(OPEN, -11) / ts_delay(OPEN, -1) - 1
    RETURN_OC_1 = ts_delay(CLOSE, -1) / ts_delay(OPEN, -1) - 1
    RETURN_CC_1 = ts_delay(CLOSE, -1) / CLOSE - 1
    RETURN_CO_1 = ts_delay(OPEN, -1) / CLOSE - 1


def _code_block_2():
    # filter后计算的代码

    # TODO 打标签应当在票池中打，还是在全A中打？
    # LABEL_OO_5 = cs_winsorize_mad(RETURN_OO_5)
    LABEL_OO_5 = cs_bucket(cs_winsorize_mad(RETURN_OO_5), 20)
    LABEL_OO_10 = cs_bucket(cs_winsorize_mad(RETURN_OO_10), 20)

    # TODO 本人尝试的pe指标处理方法，不知是否合适，欢迎指点
    # 对数市值。去极值，标准化，行业中性化。反向
    LOG_MKT_CAP_NEUT = -cs_neutralize_residual_multiple(cs_standardize_zscore(cs_winsorize_mad(LOG_MKT_CAP, 3)), CS_SW_L1, ONE)

    # pe为负已经提前过滤了
    FEATURE_00 = cs_standardize_zscore(cs_winsorize_mad(1 / pe_ratio, 3))
    # 去极值、标准化
    FEATURE_01 = -abs_(cs_rank(cs_standardize_zscore(cs_winsorize_mad(1 / pe_ratio, 3))) - 0.5)
    # 去极值、标准化、行业中性化
    FEATURE_02 = -abs_(cs_rank(cs_neutralize_residual_multiple(cs_standardize_zscore(cs_winsorize_mad(1 / pe_ratio, 3)), CS_SW_L1, ONE)) - 0.6)
    # 去极值、标准化、市值中性化
    FEATURE_03 = -abs_(cs_rank(cs_neutralize_residual_multiple(cs_standardize_zscore(cs_winsorize_mad(1 / pe_ratio, 3)), LOG_MKT_CAP_NEUT, ONE)) - 0.7)
    # 去极值、标准化、行业市值中性化
    FEATURE_04 = -abs_(cs_rank(cs_neutralize_residual_multiple(cs_standardize_zscore(cs_winsorize_mad(1 / pe_ratio, 3)), CS_SW_L1, LOG_MKT_CAP_NEUT, ONE)) - 0.7)


def code_to_string(code_block):
    source = inspect.getsource(code_block)
    raw, exprs_dict = sources_to_exprs(globals().copy(), source, safe=False)

    # 生成代码
    codes, G = ExprTool().all(exprs_dict, style='polars', template_file='template.py.j2',
                              replace=True, regroup=True, format=True,
                              date='date', asset='asset',
                              # 覆盖`CS_SW_L1 = pl.col("CS_SW_L1")`的定义为正则风格
                              alias={'CS_SW_L1': r"^sw_l1_\d+$"},
                              # 复制了需要使用的函数，还复制了最原始的表达式
                              extra_codes=(raw,))

    return codes


# ======================================
# 保存代码到指定文件
output_file = 'research/output1.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(code_to_string(_code_block_1))

# 保存代码到指定文件
output_file = 'research/output2.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(code_to_string(_code_block_2))

logger.info('转码完成')
# =======================================
# %% 生成因子
# 由于读写多，推荐放到内存盘
DATA_PATH = r'M:\data3\T1\data.parquet'
FEATURE_PATH = r'M:\data3\T1\feature.parquet'

df = pl.read_parquet(DATA_PATH)
df = df.rename({'time': 'date', 'code': 'asset', 'money': 'amount'})
print(df.columns)
# 计算收益率前，提前过滤。收益率计算时不能跳过st等信息
df = df.filter(
    pl.col('date') > datetime(2018, 1, 1),  # 过滤要测试用的数据时间范围
    pl.col('paused') == 0,  # 过滤停牌
    ~pl.col('asset').str.starts_with('68'),  # 过滤科创板
    ~pl.col('asset').str.starts_with('30'),  # 过滤创业板
    pl.col('sw_l1').is_not_null(),  # TODO 没有行业的也过滤，这会不会有问题？
)
# 准备基础数据
df = df.with_columns([
    # 后复权
    (pl.col(['open', 'high', 'low', 'close']) * pl.col('factor')).name.map(lambda x: x.upper()),
    # 成交额与成交量对数处理
    pl.col('amount').log1p().alias('LOG_AMOUNT'),
    pl.col('volume').log1p().alias('LOG_VOLUME'),
    pl.col('market_cap').log1p().alias('LOG_MKT_CAP'),
    # 添加常数列，也许回归等场景用得上
    pl.lit(1, dtype=pl.Float32).alias('ONE'),
    pl.lit(0, dtype=pl.Float32).alias('ZERO'),
    # 行业处理，由浮点改成整数
    pl.col('sw_l1', 'sw_l2', 'sw_l3').cast(pl.UInt32),
]).fill_nan(None)  # nan填充成null
logger.info('数据准备完成')

# =====================================
from research.output1 import main

df = main(df)

# 计算出来的结果需要进行部分修复，防止之后计算时出错
df = df.with_columns(pl.col('NEXT_DOJI').fill_null(False))
# st不参与后面的计算
# TODO 也可以设置只计算中证500等
df = df.filter(~pl.col('is_st'))
df = df.filter(pl.col('pe_ratio') > 0)
sw_l1 = df.select('sw_l1')
# TODO drop_first丢弃哪个字段是随机的，非常不友好，只能在行业中性化时动态修改代码
df = df.with_columns(df.to_dummies('sw_l1', drop_first=True))

from research.output2 import main

df = main(df)

# TODO 过滤掉不参与IC计算和机器学习的记录
# 过滤明天涨停或跌停
df = df.filter(~pl.col('NEXT_DOJI'))
# 将计算结果中的inf都换成null
df = df.with_columns(fill_nan(fill_infinite(cs.numeric())).name.keep())

logger.info('特征计算完成')
# =====================================
# 推荐保存到内存盘中
df.write_parquet(FEATURE_PATH)
logger.info('特征保存完成')
