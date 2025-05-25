"""
计算预期收益率，注意：全是未来数据
"""
import os
import sys
from pathlib import Path

from polars_ta.wq import purify

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import polars as pl
import polars.selectors as cs
from expr_codegen import codegen_exec
from loguru import logger


def _code_block_2():
    # 远期收益率,由于平移过,含未来数据，只能用于打标签，不能用于训练
    _OC_01 = CLOSE[-1] / OPEN[-1]
    _CC_01 = CLOSE[-1] / CLOSE
    _CO_01 = OPEN[-1] / CLOSE
    _OO_01 = OPEN[-2] / OPEN[-1]

    _OO_02 = OPEN[-3] / OPEN[-1]
    _OO_05 = OPEN[-6] / OPEN[-1]
    _OO_10 = OPEN[-11] / OPEN[-1]

    # 一期收益率
    RETURN_OC_01 = _OC_01 - 1
    RETURN_CC_01 = _CC_01 - 1
    RETURN_CO_01 = _CO_01 - 1
    RETURN_OO_01 = _OO_01 - 1

    # 算术平均
    RETURN_OO_02 = (_OO_02 - 1) / 2
    RETURN_OO_05 = (_OO_05 - 1) / 5
    RETURN_OO_10 = (_OO_10 - 1) / 10

    # 几何平均
    RETURN_OO_02 = _OO_02 ** (1 / 2) - 1
    RETURN_OO_05 = _OO_05 ** (1 / 5) - 1
    RETURN_OO_10 = _OO_10 ** (1 / 10) - 1


# =======================================
# %% 生成因子


if __name__ == '__main__':
    # 由于读写多，推荐放到内存盘，加快速度
    INPUT_PATH = r'M:\preprocessing\data2.parquet'
    # 去除停牌后的基础数据
    OUTPUT_PATH = r'M:\preprocessing\data3.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)
    print(df.columns)

    logger.info('数据准备完成')
    # =====================================

    df = codegen_exec(df, _code_block_2, over_null="partition_by")

    # 将计算结果中的inf都换成null
    df = df.with_columns(purify(cs.numeric()))

    logger.info('特征计算完成')

    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)
