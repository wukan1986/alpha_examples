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

    # TODO 由于ts_decay_linear不支持null，暂时用ts_mean代替
    from polars_ta.prefix.wq import ts_mean as ts_decay_linear  # noqa
    from polars_ta.prefix.wq import ts_mean as ts_WMA  # noqa

    RET = ts_returns(CLOSE, 1)


with open('transformer/alpha191_out.txt', 'r') as f:
    source1 = f.read()

source = inspect.getsource(_code_block_)
raw, exprs_dict = sources_to_exprs(globals().copy(), source, source1)
# 生成代码
tool = ExprTool()
codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 还复制了最原始的表达式
                    extra_codes=(raw,))

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/alpha191.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
