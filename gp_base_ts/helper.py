import time
from datetime import datetime
from typing import Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
from expr_codegen.tool import ExprTool
from loguru import logger

from gp_base_cs.base import get_fitness


def fitness_individual(a: str, b: str) -> pl.Expr:
    """个体fitness函数"""
    return pl.corr(a, b, method='pearson', ddof=0, propagate_nans=False)


def root_operator(df: pl.DataFrame):
    """强插一个根算子

    比如挖掘出的因子是
    ts_SMA(CLOSE, 10)
    ts_returns(ts_SMA(CLOSE, 20),1)

    这一步相当于在外层套一个ts_zscore，变成
    ts_zscore(ts_SMA(CLOSE, 10),120)
    ts_zscore(ts_returns(ts_SMA(CLOSE, 20),1),120)

    注意，复制到其它工具验证时，一定要记得要带上根算子

    这里只对GP_开头的因子添加根算子

    """
    from polars_ta.prefix.wq import ts_zscore  # noqa
    from polars_ta.prefix.wq import cs_mad, cs_zscore  # noqa

    def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
        df = df.sort(by=['date'])
        # ========================================
        df = df.with_columns(
            ts_zscore(pl.col(r'^GP_\d+$'), 120),
        )
        return df

    df = df.group_by('asset').map_groups(func_0_ts__asset)
    logger.warning("启用了根算子，复制到其它平台时记得手工添加")

    return df


def fitness_population(df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
    """种群fitness函数"""
    if df is None:
        return {}, {}

    # TODO 是否要强插一个根算子???
    # df = root_operator(df)

    # 将IC划分成训练集与测试集
    df_train = df.filter(pl.col('date') < split_date)
    df_valid = df.filter(pl.col('date') >= split_date)

    # 时序相关性，没有IR
    ic_train = df_train.group_by('asset').agg([fitness_individual(X, label) for X in columns]).fill_nan(None)
    ic_valid = df_valid.group_by('asset').agg([fitness_individual(X, label) for X in columns]).fill_nan(None)

    # 时序IC的多资产平均。可用来挖掘在多品种上适应的因子
    ic_train = ic_train.select(pl.when(cs.numeric().is_not_null().mean() >= 0.8).then(cs.numeric().mean()).otherwise(None))
    ic_valid = ic_valid.select(cs.numeric().mean())

    ic_train = ic_train.to_dicts()[0]
    ic_valid = ic_valid.to_dicts()[0]

    return ic_train, ic_valid


def batched_exprs(batch_id, exprs_dict, gen, label, split_date, df_input):
    """每代种群分批计算

    由于种群数大，一次性计算可能内存不足，所以提供分批计算功能，同时也为分布式计算做准备
    """
    if len(exprs_dict) == 0:
        return {}

    tool = ExprTool()
    # 表达式转脚本
    codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                        replace=False, regroup=True, format=True,
                        date='date', asset='asset')

    cnt = len(exprs_dict)
    logger.info("{}代{}批 代码 开始执行。共 {} 条 表达式", gen, batch_id, cnt)
    tic = time.perf_counter()

    globals_ = {'df_input': df_input}
    exec(codes, globals_)
    df_output = globals_['df_output']

    elapsed_time = time.perf_counter() - tic
    logger.info("{}代{}批 因子 计算完成。共用时 {:.3f} 秒，平均 {:.3f} 秒/条，或 {:.3f} 条/秒", gen, batch_id, elapsed_time, elapsed_time / cnt, cnt / elapsed_time)

    # 计算种群适应度
    ic_train, ic_valid = fitness_population(df_output, list(exprs_dict.keys()), label=label, split_date=split_date)
    logger.info("{}代{}批 适应度 计算完成", gen, batch_id)

    # 样本内外适应度提取
    new_results = {}
    for k, v in exprs_dict.items():
        v = str(v)
        new_results[v] = {'ic_train': get_fitness(k, ic_train),
                          'ic_valid': get_fitness(k, ic_valid),
                          }
    return new_results


def fill_fitness(exprs_old, fitness_results):
    """填充fitness"""
    results = []
    for k, v in exprs_old.items():
        v = str(v)
        d = fitness_results.get(v, None)
        if d is None:
            logger.debug('{} 不合法/无意义/重复 等原因，在计算前就被剔除了', v)
        else:
            s0, s1 = d['ic_train'], d['ic_valid']
            # ic要看绝对值
            s0, s1 = abs(s0), abs(s1)
            # TODO 这地方要按自己需求定制，过滤太多可能无法输出有效表达式
            if s0 == s0:  # 非空
                if s0 > 0.001:  # 样本内打分要大
                    if s0 * 0.6 < s1:  # 样本外打分大于样本内打分的70%
                        # 可以向fitness添加多个值，但长度要与weight完全一样
                        results.append((s0, s1))
                        continue
        # 可以向fitness添加多个值，但长度要与weight完全一样
        results.append((np.nan, np.nan))

    return results
