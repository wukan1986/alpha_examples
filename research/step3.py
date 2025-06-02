"""
读取指定数据文件中的多个特征，生成多个报表
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
import multiprocessing
import time

import polars as pl
from alphainspect.reports import report_html
from alphainspect.utils import with_factor_quantile
from loguru import logger

# 收益率文件
INPUT1_PATH = r'M:\preprocessing\data3.parquet'
# 特征数据文件
INPUT2_PATH = r'M:\preprocessing\data4.parquet'
# 输出目录
output = Path(r'M:\preprocessing\output')


def func(kv):
    name, factors = kv
    fwd_ret_1 = 'RETURN_OO_02'
    label = 'LABEL_OO_02'

    # 收益信息
    df1 = pl.read_parquet(INPUT1_PATH, columns=['date', 'asset', fwd_ret_1])
    # 只记录特征，收益不全
    df2 = pl.read_parquet(INPUT2_PATH, columns=['date', 'asset', 'NEXT_DOJI4', label] + factors)

    # !!! 提前排序，否则计算时会很慢
    df1 = df1.sort('date', 'asset')
    df2 = df2.sort('date', 'asset')

    for factor in factors:
        df2 = with_factor_quantile(df2, factor, quantiles=5, factor_quantile=f'_fq_{factor}')
        df2 = df2.filter(~pl.col('NEXT_DOJI4'))

    # 合并，特征数据由于过滤了了一些记录，不够用来计算收益，所以需要合并
    df = df1.join(df2, on=['date', 'asset'], how='left')
    df = df.sort('date', 'asset')

    report_html(name, factors, df, output,
                fwd_ret_1=label, quantiles=0, top_k=0, axvlines=('2020-01-01', '2024-01-01',))

    return 0


if __name__ == '__main__':
    # 1去极值标准化/2市值中性化/3行业中性化/4行业市值中性化
    factors2 = {
        "1_线性正向": ['MC_LOG', 'MC_NORM', 'MC_NEUT'],
    }
    t0 = time.perf_counter()
    logger.info('开始')
    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    # 参考CPU与内存，可以考虑在这填写合适的值，如：4、8
    with multiprocessing.Pool(2) as pool:
        print(list(map(func, factors2.items())))
    logger.info('结束')
    logger.info(f'耗时：{time.perf_counter() - t0:.2f}s')
    os.system(f'explorer.exe "{output}"')
