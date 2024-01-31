"""

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

import polars as pl

from loguru import logger

logger.info('加载数据')
df = pl.read_parquet('data/data.parquet')

logger.info('生成特征')
from codes.features import main

df = main(df)
logger.info('特征生成完成')
print(df.tail())

logger.info('保存')
df.write_parquet('data/features.parquet')
