# %%
import os
import sys

from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# %% 因子报表
import matplotlib.pyplot as plt
import polars as pl
from alphainspect.ic import create_ic2_sheet
from alphainspect.reports import create_2x2_sheet
from alphainspect.returns import create_returns_sheet
from alphainspect.utils import with_factor_quantile
from loguru import logger

logger.info('加载数据')
# 数据从内存盘中读更好
FEATURE_PATH = r'M:\data3\T1\feature.parquet'
df = pl.read_parquet(FEATURE_PATH)
print(df.columns)

factors = [
    'FEATURE_01',
    'FEATURE_02',
    'FEATURE_03',
    'FEATURE_04',
]
forward_returns = ['RETURN_CC_1', 'RETURN_OO_1', 'RETURN_OO_5', 'RETURN_OO_10', 'LABEL_OO_5', 'LABEL_OO_10', ]  # 同一因子，不同持有期对比
logger.info('开始生成报表')
create_ic2_sheet(df, factors, forward_returns)
logger.info('查看单个因子')
plt.show()

# 需展示的某一个特征
print(list(enumerate(factors)))
factor = factors[int(input(f'输入序号：'))]
fwd_ret_1 = 'RETURN_OO_1'
forward_return = 'RETURN_OO_10'
period = 10
axvlines = ('2023-01-01',)

df = with_factor_quantile(df, factor, quantiles=10)
forward_returns = ['RETURN_CC_1', 'RETURN_OO_1', 'RETURN_OO_5', 'RETURN_OO_10', ]
create_returns_sheet(df, factor, forward_returns)

create_2x2_sheet(df, factor, forward_return, fwd_ret_1, period=period, axvlines=axvlines)
logger.info('完成')

plt.show()

# %%
