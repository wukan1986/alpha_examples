"""
准备基本测试数据演示

必须有date, asset两个字段

由于数据准备是一个比较耗时耗力的过程，为了能快速演示，所以这里使用了生成的数据
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

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

_N = 250 * 15
_K = 500

asset = [f's_{i:04d}' for i in range(_K)]
date = pd.date_range('2015-1-1', periods=_N)

df = pd.DataFrame({
    # 原始价格
    'open': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'high': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'low': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'close': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'vwap': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    'volume': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    # TODO 昨收价。从交易所查询得来。注意：由于除权除息的原因，昨收价不等于昨天的收盘价
    'pre_close': np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0).reshape(-1),
    # TODO 后复权因子。后复权因子不会历史数据不会变化
    'factor': np.ones(shape=(_N, _K)).reshape(-1),
    # TODO 这只是为了制造长度不同的数据而设计
    "FILTER": np.tri(_N, _K, k=100).reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset'])).reset_index()

# 向脚本输入数据
df = pl.from_pandas(df)
# 数据长度不同
df = df.filter(pl.col('FILTER') == 1).drop(columns=['FILTER'])

# 复权
df = df.with_columns([
    (pl.col(['open', 'high', 'low', 'close']) * pl.col('factor')).name.map(lambda x: x.upper()),
    pl.col('volume').alias('VOLUME'),
    pl.col('vwap').alias('VWAP'),
])

# 打标签，由于标签常用到，所以这里提前打在基础数据中，用户按自己需求调整
from codes.labels import main

logger.info('生成特征')
df = main(df)
logger.info('特征生成完成')
print(df.tail())

# 数据压缩，f64改f32。
# 注意：talib只支持64位，如果做完talib计算后，再转f32更优
df = df.select(pl.all().shrink_dtype())
df = df.shrink_to_fit()

print(df.tail())

logger.info('保存')
df.write_parquet('data/data.parquet')
