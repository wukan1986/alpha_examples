# %%
import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())

# %% 表达式转换
import inspect

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    SMA_010 = cs_standardize_zscore(cs_winsorize_3sigma(CLOSE / ts_mean(CLOSE, 10), 3))
    SMA_020 = cs_standardize_zscore(cs_winsorize_3sigma(CLOSE / ts_mean(CLOSE, 20), 3))


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

# %% 生成因子
import polars as pl

df = pl.read_parquet('data/data.parquet')
logger.info('数据加载完成')

_globals = {'df_input': df}
exec(codes, _globals)
df = _globals['df_output']
logger.info('因子计算完成')
df.tail()

# %% 因子报表
import matplotlib.pyplot as plt
from alphainspect.reports import create_3x2_sheet
from alphainspect.utils import with_factor_quantile
from alphainspect.ic import plot_ic_hist

factor = 'SMA_010'
fwd_ret_1 = 'RETURN_OO_1'
forward_return = 'RETURN_OO_5'
period = 5
axvlines = ('2020-01-01',)

logger.info('开始生成报表')
# 画ic的直方图函数，也可以用来画普通数值
plot_ic_hist(df, factor)
# plt.show()

df = with_factor_quantile(df, factor, quantiles=10)
create_3x2_sheet(df, factor, forward_return, fwd_ret_1, period=period, axvlines=axvlines)
logger.info('报表已生成')

plt.show()

# %%
