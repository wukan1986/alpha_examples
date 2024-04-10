"""
读取指定数据文件中的多个特征，生成多个报表
"""
import os
import sys
import time
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import multiprocessing

import matplotlib.pyplot as plt
import polars as pl
from alphainspect.reports import create_1x3_sheet, fig_to_img, html_template
from alphainspect.utils import with_factor_quantile, with_quantile_tradable
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa

# 特征数据文件
FEATURE_PATH = r'M:\data3\T1\feature.parquet'
# 输出目录
output = Path(r'M:\data3\T1\output')

output.mkdir(parents=True, exist_ok=True)


def func(kv):
    name, factors = kv
    fwd_ret_1 = 'RETURN_OO_1'
    forward_return = 'LABEL_OO_5'
    period = 5
    axvlines = ('2023-01-01',)
    quantiles = 10

    df = pl.read_parquet(FEATURE_PATH, columns=['date', 'asset', 'NEXT_DOJI'] + [forward_return, fwd_ret_1] + factors, use_pyarrow=True)
    for factor in factors:
        df = with_factor_quantile(df, factor, quantiles=quantiles, factor_quantile=f'_fq_{factor}')
        # 明日涨跌停分到-1组
        df = with_quantile_tradable(df, f'_fq_{factor}', 'NEXT_DOJI')

    tbl = {}
    imgs = []
    for factor in factors:
        fig, ic_dict, hist_dict, df_cum_ret = create_1x3_sheet(df, factor, forward_return, fwd_ret_1,
                                                               period=period,
                                                               factor_quantile=f'_fq_{factor}',
                                                               figsize=(12, 3),
                                                               axvlines=axvlines)
        s1 = df_cum_ret.iloc[-1]
        s2 = {'monotonic': np.sign(s1.diff()).sum()}
        s3 = pd.Series(s2 | ic_dict | hist_dict)
        tbl[factor] = pd.concat([s1, s3])
        imgs.append(fig_to_img(fig))

    # 各指标柱状图
    tbl = pd.DataFrame(tbl)
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    ax1 = tbl.iloc[:quantiles].plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.legend(loc='upper left')
    imgs.insert(0, fig_to_img(fig))

    # 表格
    txt1 = tbl.T.to_html(float_format=lambda x: format(x, '.4f'))
    # 图
    txt2 = '\n'.join(imgs)
    tpl = html_template.replace('{{body}}', f'{txt1}\n{txt2}')

    with open(str(output / f'{name}.html'), "w", encoding="utf-8") as f:
        f.write(tpl)

    return 0


if __name__ == '__main__':
    # 1去极值标准化/2市值中性化/3行业中性化/4行业市值中性化
    factors2 = {
        "1_线性正向": ['F_11', 'F_12', 'F_13', 'F_14', ],
        "2_参数对比": [
            # 'F_005',
            # 'F_010',
            # 'F_015',
            # 'F_020',
            # 'F_025',
            'F_030',
            'F_035',
            'F_040',
            'F_045',
            'F_050',
            'F_055',
            # 'F_060',
            # 'F_065',
        ],
    }
    t0 = time.perf_counter()
    logger.info('开始')
    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    # 参考CPU与内存，可以考虑在这填写合适的值，如：4、8
    with multiprocessing.Pool(2) as pool:
        print(list(pool.map(func, factors2.items())))
    logger.info('结束')
    logger.info(f'耗时：{time.perf_counter() - t0:.2f}s')
    os.system(f'explorer.exe "{output}"')
