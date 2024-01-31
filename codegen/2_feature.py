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

from expr_codegen.expr import string_to_exprs
from expr_codegen.tool import ExprTool

# 导入OPEN等特征
from sympy_define import *  # noqa


def _expr_code():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    ROCP_020 = ts_returns(CLOSE, 20)
    ROCP_040 = ts_returns(CLOSE, 40)
    ROCP_060 = ts_returns(CLOSE, 60)

    VR_020 = ts_std_dev(ts_log_diff(CLOSE, 1), 20)
    VR_040 = ts_std_dev(ts_log_diff(CLOSE, 1), 40)
    VR_060 = ts_std_dev(ts_log_diff(CLOSE, 1), 60)

    SMA_020 = ts_mean(CLOSE, 20)
    SMA_040 = ts_mean(CLOSE, 40)
    SMA_060 = ts_mean(CLOSE, 60)


# 读取源代码，转成字符串
source = inspect.getsource(_expr_code)
exprs_txt = []
# 将字符串转成表达式，与streamlit中效果一样
exprs_src = string_to_exprs('\n'.join([source] + exprs_txt), globals().copy())

# 生成代码
tool = ExprTool()
codes, G = tool.all(exprs_src, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 还复制了最原始的表达式
                    )

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/features.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
