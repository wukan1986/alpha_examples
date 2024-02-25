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
# 表达式转换

# 导入OPEN等特征
from sympy_define import *  # noqa
from alphainspect.reports import ipynb_to_html


def func(factor):
    # 特征数据文件
    FEATURE_PATH = r'M:\data3\T1\feature.parquet'
    # 输出目录
    output = Path('research/output')

    output.mkdir(parents=True, exist_ok=True)
    ret_code = ipynb_to_html('research/template.ipynb',
                             output=str(output / f'{factor}.html'),
                             no_input=True,
                             no_prompt=False,
                             open_browser=False,
                             # 以下参数转成环境变量自动变成大写
                             PWD=os.getcwd(),
                             FEATURE_PATH=FEATURE_PATH,
                             FACTOR=factor,
                             fwd_ret_1='RETURN_OO_1',
                             forward_return='RETURN_OO_5',
                             period=5)

    return ret_code


if __name__ == '__main__':
    import multiprocessing

    # 没必要设置太大，因为部分计算使用的polars多线程，会将CPU跑满
    factors = [
        'FEATURE_01',
        'FEATURE_02',
        'FEATURE_03',
        # 'FEATURE_04',
        # 'FEATURE_05',
    ]
    with multiprocessing.Pool() as pool:
        print(list(pool.map(func, factors)))
