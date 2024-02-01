"""
演示如何生成标签生成代码

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
# 表达式转换
import inspect

from expr_codegen.codes import sources_to_asts
from expr_codegen.expr import dict_to_exprs
from expr_codegen.tool import ExprTool
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 这里用未复权的价格更合适
    # 今日涨停或跌停
    SMA_010 = CLOSE / ts_mean(CLOSE, 10)
    SMA_020 = CLOSE / ts_mean(CLOSE, 20)


# 读取源代码，转成字符串
source = inspect.getsource(_code_block_)
raw, assigns = sources_to_asts(source)
assigns_dict = dict_to_exprs(assigns, globals().copy())

# 生成代码
tool = ExprTool()
codes, G = tool.all(assigns_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 复制了需要使用的函数，还复制了最原始的表达式
                    extra_codes=(raw, _code_block_,))

# print(codes)
logger.info('转码完成')
# 保存代码到指定文件，在Notebook中将会使用它
output_file = 'research/output.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
# ====================
# 因子报表
from alphainspect.reports import ipynb_to_html

logger.info('生成报表')

factor = 'SMA_020'
ipynb_to_html('research/template.ipynb',
              output=f'research/{factor}.html',
              no_input=True,
              no_prompt=False,
              open_browser=True,
              # 以下参数转成环境变量自动变成大写
              pwd=os.getcwd(),
              factor=factor,
              fwd_ret_1='RETURN_OO_1',
              forward_return='RETURN_OO_5',
              period=5)

logger.info('浏览器已关闭')
