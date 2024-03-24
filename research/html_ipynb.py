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
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa
from alphainspect.reports import ipynb_to_html

# 特征数据文件
FEATURE_PATH = r'M:\data3\T1\feature.parquet'
# 输出目录
output = Path(r'M:\data3\T1\output')

output.mkdir(parents=True, exist_ok=True)


def func(kv):
    name, factors = kv

    ret_code = ipynb_to_html('research/template.ipynb',
                             output=str(output / f'{name}.html'),
                             no_input=True,
                             no_prompt=True,
                             open_browser=False,
                             # 以下参数转成环境变量自动变成大写
                             PWD=os.getcwd(),
                             FEATURE_PATH=FEATURE_PATH,
                             FACTORS=factors,
                             fwd_ret_1='RETURN_OO_1',
                             forward_return='LABEL_OO_5',
                             period=5)

    return ret_code


if __name__ == '__main__':
    # 1去极值标准化/2市值中性化/3行业中性化/4行业市值中性化
    factors2 = {
        "1_线性正向": ['FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14', ],
        "2_线性反向": ['FEATURE_21', 'FEATURE_22', 'FEATURE_23', 'FEATURE_24', ],
        "3_非线反向": ['FEATURE_31', 'FEATURE_32', 'FEATURE_33', 'FEATURE_34', ],
        # "4_非线正向": ['FEATURE_41', 'FEATURE_42', 'FEATURE_43', 'FEATURE_44', ],
    }
    t0 = time.perf_counter()
    logger.info('开始')
    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    # 参考CPU与内存，可以考虑在这填写合适的值，如：4、8
    with multiprocessing.Pool(4) as pool:
        print(list(pool.map(func, factors2.items())))
    logger.info('结束')
    logger.info(f'耗时：{time.perf_counter() - t0:.2f}s')
    os.system(f'explorer.exe "{output}"')
