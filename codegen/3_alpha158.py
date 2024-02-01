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

from expr_codegen.codes import sources_to_asts
from expr_codegen.expr import dict_to_exprs
from expr_codegen.tool import ExprTool

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 以下因子来原于qlib项目的Alpha158
    # https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

    # 标签
    LABEL0 = ts_delay(VWAP, -2) / ts_delay(VWAP, -1) - 1

    # 特征

    # kbar
    KMID = (CLOSE - OPEN) / OPEN
    KLEN = (HIGH - LOW) / OPEN
    KMID2 = (CLOSE - OPEN) / (HIGH - LOW + 1e-12)
    KUP = (HIGH - max_(OPEN, CLOSE)) / OPEN
    KUP2 = (HIGH - max_(OPEN, CLOSE)) / (HIGH - LOW + 1e-12)
    KLOW = (min_(OPEN, CLOSE) - LOW) / OPEN
    KLOW2 = (min_(OPEN, CLOSE) - LOW) / (HIGH - LOW + 1e-12)
    KSFT = (2 * CLOSE - HIGH - LOW) / OPEN
    KSFT2 = (2 * CLOSE - HIGH - LOW) / (HIGH - LOW + 1e-12)

    # price
    OPEN0 = OPEN / CLOSE
    HIGH0 = HIGH / CLOSE
    LOW0 = LOW / CLOSE

    # volume

    # rolling
    # BETA
    # RSQR
    # RESI


# 读取源代码，转成字符串
source = inspect.getsource(_code_block_)
source2 = """
# 只有在sources_to_asts之前才能添加三元表达式
_TEST1 = OPEN>CLOSE?OPEN:CLOSE
_TEST2 = (OPEN>CLOSE)*-1
_TEST3 = (OPEN==CLOSE)*-1
"""
raw, assigns = sources_to_asts(source, source2)

for i in (1, 2, 3, 4):
    assigns[f'OPEN{i}'] = f'ts_delay(OPEN, {i}) / CLOSE'
    assigns[f'HIGH{i}'] = f'ts_delay(HIGH, {i}) / CLOSE'
    assigns[f'LOW{i}'] = f'ts_delay(LOW, {i}) / CLOSE'
    assigns[f'VOLUME{i}'] = f'ts_delay(VOLUME, {i}) / (VOLUME + 1e-12)'

for i in (5, 10, 20, 30, 60):
    # 注意：qlib的roc与talib中的表示方法不同
    assigns[f'ROC{i}'] = f'ts_delay(CLOSE, {i}) / CLOSE'
    assigns[f'MA{i}'] = f'ts_mean(CLOSE, {i}) / CLOSE'
    assigns[f'STD{i}'] = f'ts_std_dev(CLOSE, {i}) / CLOSE'
    assigns[f'MAX{i}'] = f'ts_max(CLOSE, {i}) / CLOSE'
    assigns[f'MIN{i}'] = f'ts_min(CLOSE, {i}) / CLOSE'
    assigns[f'QTLU{i}'] = f'ts_percentage(CLOSE, {i}, 0.8) / CLOSE'
    assigns[f'QTLD{i}'] = f'ts_percentage(CLOSE, {i}, 0.2) / CLOSE'
    assigns[f'RANK{i}'] = f'ts_rank(CLOSE, {i}) / CLOSE'
    assigns[f'RSV{i}'] = f'ts_RSV(HIGH, LOW, CLOSE, {i})'
    assigns[f'CORR{i}'] = f'ts_corr(CLOSE, log1p(VOLUME), {i})'
    assigns[f'CNTP{i}'] = f'ts_mean(CLOSE > ts_delay(CLOSE, 1), {i})'
    assigns[f'CNTN{i}'] = f'ts_mean(CLOSE < ts_delay(CLOSE, 1), {i})'
    assigns[f'CNTD{i}'] = f'CNTP{i}-CNTN{i}'
    assigns[f'SUMP{i}'] = f'ts_sum(max_(CLOSE - ts_delay(CLOSE, 1), 0), {i}) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), {i}) + 1e-12)'
    assigns[f'SUMN{i}'] = f'ts_sum(max_(ts_delay(CLOSE, 1) - CLOSE, 0), {i}) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), {i}) + 1e-12)'
    assigns[f'SUMD{i}'] = f'SUMP{i}-SUMN{i}'
    assigns[f'VMA{i}'] = f'ts_mean(VOLUME, {i}) / (VOLUME + 1e-12)'
    assigns[f'VSTD{i}'] = f'ts_std_dev(VOLUME, {i}) / (VOLUME + 1e-12)'
    assigns[f'WVMA{i}'] = f'ts_std_dev(abs_(ts_returns(CLOSE, 1)) * VOLUME, {i}) / (ts_mean(abs_(ts_returns(CLOSE, 1)) * VOLUME, {i}) + 1e-12)'
    assigns[f'VSUMP{i}'] = f'ts_sum(max_(VOLUME - ts_delay(VOLUME, 1), 0), {i}) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), {i}) + 1e-12)'
    assigns[f'VSUMN{i}'] = f'ts_sum(max_(ts_delay(VOLUME, 1) - VOLUME, 0), {i}) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), {i}) + 1e-12)'
    assigns[f'VSUMD{i}'] = f'VSUMP{i}-VSUMN{i}'

assigns = dict(sorted(assigns.items(), key=lambda x: x[0]))
assigns_dict = dict_to_exprs(assigns, globals().copy())
# 生成代码
tool = ExprTool()
codes, G = tool.all(assigns_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 还复制了最原始的表达式
                    extra_codes=())

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/alpha158.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
