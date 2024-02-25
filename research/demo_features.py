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


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 这里用未复权的价格更合适
    DOJI = four_price_doji(OPEN, HIGH, LOW, CLOSE)
    NEXT_DOJI = ts_delay(DOJI, -1)

    # 远期收益率
    RETURN_OO_1 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1) - 1
    RETURN_OO_2 = ts_delay(OPEN, -3) / ts_delay(OPEN, -1) - 1
    RETURN_OO_5 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1) - 1
    RETURN_OC_1 = ts_delay(OPEN, -1) / ts_delay(CLOSE, -1) - 1
    RETURN_CC_1 = ts_delay(CLOSE, -1) / CLOSE - 1
    RETURN_CO_1 = ts_delay(OPEN, -1) / CLOSE - 1

    # 多个特征，用来进行比较
    FEATURE_01 = -abs_(cs_standardize_zscore(ts_NATR(HIGH, LOW, CLOSE, 5)))
    FEATURE_02 = -abs_(cs_standardize_zscore(ts_NATR(HIGH, LOW, CLOSE, 10)))
    FEATURE_03 = -abs_(cs_standardize_zscore(ts_NATR(HIGH, LOW, CLOSE, 20)))


# 读取源代码，转成字符串
source = inspect.getsource(_code_block_)
raw, exprs_dict = sources_to_exprs(globals().copy(), source)

# 生成代码
tool = ExprTool()
codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 复制了需要使用的函数，还复制了最原始的表达式
                    extra_codes=(raw, _code_block_,))

# print(codes)
logger.info('转码完成')
# 保存代码到指定文件，在Notebook中将会使用它
output_file = 'research/output.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
# ====================
# %% 生成因子
# 由于读写多，推荐放到内存盘
DATA_PATH = r'M:\data3\T1\data.parquet'
FEATURE_PATH = r'M:\data3\T1\feature.parquet'

import polars as pl

df = pl.read_parquet(DATA_PATH)
df = df.rename({'time': 'date', 'code': 'asset', 'money': 'amount'})
# 过滤要测试用的数据
df = df.filter(pl.col('date') > datetime(2018, 1, 1))
df = df.with_columns([
    (pl.col(['open', 'high', 'low', 'close']) * pl.col('factor')).name.map(lambda x: x.upper()),
])
df = df.fill_nan(None)
logger.info('数据加载完成')

from research.output import main

df: pl.DataFrame = main(df)
# 明天涨停或跌停，过滤掉
df = df.filter(~pl.col('NEXT_DOJI'))

logger.info('特征计算完成')
# 推荐保存到内存盘中
df.write_parquet(FEATURE_PATH)
logger.info('特征保存完成')
