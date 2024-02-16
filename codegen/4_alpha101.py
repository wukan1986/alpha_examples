import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import inspect

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # adv{d} = average daily dollar volume for the past d days
    ADV5 = ts_mean(AMOUNT, 5)
    ADV10 = ts_mean(AMOUNT, 10)
    ADV15 = ts_mean(AMOUNT, 15)
    ADV20 = ts_mean(AMOUNT, 20)
    ADV30 = ts_mean(AMOUNT, 30)
    ADV40 = ts_mean(AMOUNT, 40)
    ADV50 = ts_mean(AMOUNT, 50)
    ADV60 = ts_mean(AMOUNT, 60)
    ADV81 = ts_mean(AMOUNT, 81)
    ADV120 = ts_mean(AMOUNT, 120)
    ADV150 = ts_mean(AMOUNT, 150)
    ADV180 = ts_mean(AMOUNT, 180)


with open('transformer/alpha101_out.txt', 'r') as f:
    source1 = f.read()

# 读取源代码，转成字符串
source = inspect.getsource(_code_block_)
raw, exprs_dict = sources_to_exprs(globals().copy(), source, source1)

# 生成代码
tool = ExprTool()
codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 还复制了最原始的表达式
                    extra_codes=())

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/alpha101.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
