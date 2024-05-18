"""
读取指定数据文件中的多个特征，生成多个报表
"""

import polars as pl
from expr_codegen.tool import codegen_exec
from lightbt import LightBT, warmup
from lightbt.callbacks import commission_by_value
from lightbt.enums import order_outside_dt, SizeType
from lightbt.signals import orders_daily
from lightbt.stats import total_equity
from lightbt.utils import Timer, groupby
from matplotlib import pyplot as plt

# 导入OPEN等特征
from sympy_define import *  # noqa

# 收益率文件
INPUT1_PATH = r'M:\data3\T1\feature1.parquet'
# 特征数据文件
INPUT2_PATH = r'M:\data3\T1\feature2.parquet'
# 行情信息
df1 = pl.read_parquet(INPUT1_PATH, columns=['date', 'asset', 'OPEN', 'CLOSE'], use_pyarrow=True)
# 特征
df2 = pl.read_parquet(INPUT2_PATH, columns=['date', 'asset', 'DOJI', 'F_11'], use_pyarrow=True)
df2 = df2.filter(pl.col('F_11').is_not_null(), ~pl.col('DOJI'))


def __code_block_1():
    FACTOR = ts_delay(F_11, 1)


df2 = codegen_exec(df2, __code_block_1)
# 由于因子值可能重复，导致前10选出的值是随机的，所以这里还多加了按代码排序，实际情况可能还要考虑其它指标
df2 = df2.group_by('date').map_groups(lambda x: x.sort(['FACTOR', 'asset'], descending=[True, False]).head(10))
df = df1.join(df2, on=['date', 'asset'], how='left')
del df1
del df2

df = df.select(pl.col('date'), pl.col('asset'),
               pl.col('OPEN').alias('fill_price'), pl.col('CLOSE').alias('last_price'),
               pl.col('FACTOR').is_not_null().cast(pl.Int64).alias('size'))
df = df.to_pandas()
df['size_type'] = SizeType.TargetValueScale

df.to_parquet('tmp.parquet')
df = pd.read_parquet('tmp.parquet')
_K = df['asset'].nunique()
_N = df['date'].nunique()
asset = df['asset'].unique()

config = pd.DataFrame({'asset': asset, 'mult': 1.0, 'margin_ratio': 1.0,
                       'commission_ratio': 0.0005, 'commission_fn': commission_by_value})

# %% 热身
print('warmup:', warmup())

# %% 初始化
bt = LightBT(init_cash=10000 * 100,  # 初始资金
             positions_precision=1.0,
             max_trades=_N * _K * 2 // 1,  # 反手占两条记录，所以预留2倍空间比较安全
             max_performances=_N * _K)

# %% 配置资产信息
with Timer():
    bt.setup(config)

# %% 资产转换，只做一次即可
df['asset'] = df['asset'].map(bt.mapping_asset_int)
df = df.sort_values(by=['date', 'asset'])
bt.run_bars(groupby(orders_daily(df, sort=False), by='date', dtype=order_outside_dt))
trades = bt.trades(return_all=True)
print(trades)
perf = bt.performances(return_all=True)

equity = total_equity(perf)['equity']
print(equity.tail())
equity.plot()
plt.show()
