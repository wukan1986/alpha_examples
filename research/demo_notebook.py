# %%
import os
import sys

from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# %%
from loguru import logger

import polars as pl

# %% 因子报表
import matplotlib.pyplot as plt
from alphainspect.reports import create_2x2_sheet
from alphainspect.utils import with_factor_quantile
from alphainspect.ic import plot_ic_hist

# 数据从内存盘中读更好
FEATURE_PATH = r'M:\data3\T1\feature.parquet'
# 需展示的特征
factor = 'FEATURE_01'
fwd_ret_1 = 'RETURN_OO_1'
forward_return = 'RETURN_OO_5'
period = 5
axvlines = ('2023-01-01',)

logger.info('加载数据')
df = pl.read_parquet(FEATURE_PATH)

logger.info('开始生成报表')
# 画ic的直方图函数，也可以用来画普通数值
plot_ic_hist(df, factor)

df = with_factor_quantile(df, factor, quantiles=10)
create_2x2_sheet(df, factor, forward_return, fwd_ret_1, period=period, axvlines=axvlines)
logger.info('报表已生成')

plt.show()

# %%
