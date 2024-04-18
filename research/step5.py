# %%
import os
import sys
from datetime import datetime
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
# ===============
# %%
import polars as pl
from matplotlib import pyplot as plt

from alphainspect.ic import create_ic2_sheet
from alphainspect.selection import drop_above_corr_thresh
from alphainspect.utils import select_by_suffix

FEATURE_PATH = r'M:\data3\T1\feature.parquet'
df_output = pl.read_parquet(FEATURE_PATH)

# x = df_output.filter(pl.col('date') == datetime(2024, 4, 12))
# %%
period = 5
axvlines = ('2024-01-01',)

# 考察因子
factors = sorted(list(filter(lambda x: x.startswith('A_'), df_output.columns)))

forward_returns = ['LABEL_OO_5']  # 同一因子，不同持有期对比
df_ic = create_ic2_sheet(df_output, factors, forward_returns)

df_pa = select_by_suffix(df_ic, '__LABEL_OO_5')
cols_to_drop, above_thresh_pairs = drop_above_corr_thresh(df_pa.to_pandas(), thresh=0.8)
# 需要剔除的因子
print('需要剔除的因子:')
print(sorted(cols_to_drop))
print(above_thresh_pairs)
print('低相关因子:')
print(sorted(list(set(factors) - set(cols_to_drop))))

plt.show()
