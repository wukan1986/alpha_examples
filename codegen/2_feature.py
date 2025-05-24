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

    ROCP_020 = ts_returns(CLOSE, 20)
    ROCP_040 = ts_returns(CLOSE, 40)
    ROCP_060 = ts_returns(CLOSE, 60)

    VR_020 = ts_std_dev(ts_log_diff(CLOSE, 1), 20)
    VR_040 = ts_std_dev(ts_log_diff(CLOSE, 1), 40)
    VR_060 = ts_std_dev(ts_log_diff(CLOSE, 1), 60)

    SMA_020 = ts_mean(CLOSE, 20)
    SMA_040 = ts_mean(CLOSE, 40)
    SMA_060 = ts_mean(CLOSE, 60)


codegen_exec(None, _code_block_, output_file='codes/features.py', over_null="partition_by")
