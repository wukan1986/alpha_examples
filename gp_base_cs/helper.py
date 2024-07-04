import time
from datetime import datetime
from typing import Sequence

import numpy as np
import polars as pl
from expr_codegen.tool import ExprTool
from loguru import logger
from polars import selectors as cs

from gp_base_cs.base import get_fitness


def fitness_individual(a: str, b: str) -> pl.Expr:
    """个体fitness函数"""
    # 这使用的是rank_ic
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


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

    def func_0_cs__date(df: pl.DataFrame) -> pl.DataFrame:
        # ========================================
        df = df.with_columns(
            cs_zscore(cs_mad(pl.col(r'^GP_\d+$'))),
        )
        return df

    df = df.group_by('date').map_groups(func_0_cs__date)
    logger.warning("启用了根算子，复制到其它平台时记得手工添加")

    return df


def fitness_population(df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
    """种群fitness函数"""
    if df is None:
        return {}, {}, {}, {}

    # TODO 是否要强插一个根算子???
    # df = root_operator(df)

    df = df.group_by('date').agg(
        [fitness_individual(X, label) for X in columns]
    ).sort(by=['date']).fill_nan(None)
    # 将IC划分成训练集与测试集
    df_train = df.filter(pl.col('date') < split_date)
    df_valid = df.filter(pl.col('date') >= split_date)

    # TODO 有效数不足，生成的意义不大，返回null, 而适应度第0位是nan时不加入名人堂
    # cs.numeric().count() / cs.numeric().len() >= 0.5
    # cs.numeric().count() >= 30
    ic_train = df_train.select(pl.when(cs.numeric().is_not_null().mean() >= 0.5).then(cs.numeric().mean()).otherwise(None))
    ic_valid = df_valid.select(cs.numeric().mean())
    ir_train = df_train.select(cs.numeric().mean() / cs.numeric().std(ddof=0))
    ir_valid = df_valid.select(cs.numeric().mean() / cs.numeric().std(ddof=0))

    ic_train = ic_train.to_dicts()[0]
    ic_valid = ic_valid.to_dicts()[0]
    ir_train = ir_train.to_dicts()[0]
    ir_valid = ir_valid.to_dicts()[0]

    return ic_train, ic_valid, ir_train, ir_valid


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
    ic_train, ic_valid, ir_train, ir_valid = fitness_population(df_output, list(exprs_dict.keys()), label=label, split_date=split_date)
    logger.info("{}代{}批 适应度 计算完成", gen, batch_id)

    # 样本内外适应度提取
    new_results = {}
    for k, v in exprs_dict.items():
        v = str(v)
        new_results[v] = {'ic_train': get_fitness(k, ic_train),
                          'ic_valid': get_fitness(k, ic_valid),
                          'ir_train': get_fitness(k, ir_train),
                          'ir_valid': get_fitness(k, ir_valid),
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
            s0, s1, s2, s3 = d['ic_train'], d['ic_valid'], d['ir_train'], d['ir_valid']
            # ic要看绝对值
            s0, s1, s2, s3 = abs(s0), abs(s1), s2, s3
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
