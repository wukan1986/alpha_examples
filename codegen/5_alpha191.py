import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================

from expr_codegen import codegen_exec


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子
    RET = ts_returns(CLOSE, 1)


with open('transformer/alpha191_out.txt', 'r') as f:
    source1 = f.read()

codegen_exec(None, _code_block_, source1, output_file='codes/alpha191.py', over_null="partition_by")
