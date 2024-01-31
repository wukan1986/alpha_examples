"""
!!! 注意：标签是未来数据，机器学习训练时只能做y,不能做X
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
import inspect

from expr_codegen.expr import string_to_exprs
from expr_codegen.tool import ExprTool

# 导入OPEN等特征
from sympy_define import *  # noqa


def cs_label(cond, x, q=20):
    """表达式太长，可自己封装一下。tool.all中指定extra_codes可以自动复制到目标代码中

    注意：名字需要考虑是否设置前缀`ts_`、`cs_`
    内部函数前缀要统一，否则生成的代码混乱。
    如cs_label与内部的cs_bucket、cs_winsorize_quantile是统一的
    """
    return if_else(cond, None, cs_bucket(cs_winsorize_quantile(x, 0.01, 0.99), q))


def _expr_code():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 这里用未复权的价格更合适
    # 今日涨停或跌停
    DOJI = four_price_doji(OPEN, HIGH, LOW, CLOSE)
    # 明日涨停或跌停
    NEXT_DOJI = ts_delay(DOJI, -1)

    # 远期收益率
    RETURN_CC_1 = ts_delay(CLOSE, -1) / CLOSE - 1
    RETURN_CO_1 = ts_delay(OPEN, -1) / CLOSE - 1
    RETURN_OC_1 = ts_delay(OPEN, -1) / ts_delay(CLOSE, -1) - 1
    RETURN_OO_1 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1) - 1
    RETURN_OO_5 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1) - 1

    # 标签
    LABEL_CC_1 = cs_label(DOJI, RETURN_CC_1, 20)
    LABEL_CO_1 = cs_label(DOJI, RETURN_CO_1, 20)
    LABEL_OC_1 = cs_label(NEXT_DOJI, RETURN_OC_1, 20)
    LABEL_OO_1 = cs_label(NEXT_DOJI, RETURN_OO_1, 20)
    LABEL_OO_5 = cs_label(NEXT_DOJI, RETURN_OO_5, 20)


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
                    # 复制了需要使用的函数，还复制了最原始的表达式
                    extra_codes=(_expr_code, cs_label,))

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/labels.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
