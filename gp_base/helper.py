import copy
import time
from datetime import datetime
from typing import Sequence, Dict

import numpy as np
import polars as pl
from expr_codegen.codes import sources_to_exprs
from expr_codegen.expr import is_meaningless
from expr_codegen.tool import ExprTool
from loguru import logger
from polars import selectors as cs
from sympy import preorder_traversal


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)

    converter = {
        'fsub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'fdiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'fmul': lambda *args_: "Mul({},{})".format(*args_),
        'fadd': lambda *args_: "Add({},{})".format(*args_),
        'fmax': lambda *args_: "max_({},{})".format(*args_),
        'fmin': lambda *args_: "min_({},{})".format(*args_),

        'isub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'idiv': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'imul': lambda *args_: "Mul({},{})".format(*args_),
        'iadd': lambda *args_: "Add({},{})".format(*args_),
        'imax': lambda *args_: "max_({},{})".format(*args_),
        'imin': lambda *args_: "min_({},{})".format(*args_),
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string.
    """
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def is_invalid(e, pset, ret_type):
    if _invalid_atom_infinite(e):
        return True
    if _invalid_number_type(e, pset, ret_type):
        return True

    return False


def _invalid_atom_infinite(e):
    """无效。单元素。无穷大或无穷小"""
    # 根是单元素，直接返回
    if e.is_Atom:
        return True
    # 有无限值
    for node in preorder_traversal(e):
        if node.is_infinite:
            return True
    return False


def _invalid_number_type(e, pset, ret_type):
    """检查参数类型"""
    # 可能导致结果为1，然后当成float去别处计算
    for node in preorder_traversal(e):
        if not node.is_Function:
            continue
        if hasattr(node, 'name'):
            node_name = node.name
        else:
            node_name = str(node.func)
        prim = pset.mapping.get(node_name, None)
        if prim is None:
            continue
        for i, arg in enumerate(prim.args):
            if issubclass(arg, ret_type):  # 此处非常重要
                if node.args[i].is_Number:
                    return True
            elif issubclass(arg, int):
                # 应当是整数，结果却是浮点
                if node.args[i].is_Float:
                    return True
            elif issubclass(arg, float):
                pass
    return False


def get_fitness(name: str, kv: Dict[str, float]) -> float:
    return kv.get(name, False) or float('nan')


def print_population(population, globals_):
    """打印种群"""
    exprs_dict = population_to_exprs(population, globals_)
    for (k, v), i in zip(exprs_dict.items(), population):
        print(f'{k}', '\t', i.fitness, '\t', v, '\t<--->\t', i)


def population_to_exprs(population, globals_):
    """群体转表达式"""
    if len(population) == 0:
        return {}
    sources = [f'GP_{i:04d}={stringify_for_sympy(expr)}' for i, expr in enumerate(population)]
    raw, exprs_dict = sources_to_exprs(globals_, '\n'.join(sources))
    return exprs_dict


def filter_exprs(exprs_dict, pset, RET_TYPE, fitness_results):
    # 清理重复表达式，通过字典特性删除
    exprs_dict = {v: k for k, v in exprs_dict.items()}
    exprs_dict = {v: k for k, v in exprs_dict.items()}
    # 清理非法表达式
    exprs_dict = {k: v for k, v in exprs_dict.items() if not is_invalid(v, pset, RET_TYPE)}
    # 清理无意义表达式
    exprs_dict = {k: v for k, v in exprs_dict.items() if not is_meaningless(v)}
    # 历史表达式不再重复计算
    before_len = len(exprs_dict)
    exprs_dict = {k: v for k, v in exprs_dict.items() if str(v) not in fitness_results}
    after_len = len(exprs_dict)
    logger.info('剔除历史已经计算过的适应度，数量由 {} -> {}', before_len, after_len)
    return exprs_dict


def fitness_individual(a: str, b: str) -> pl.Expr:
    """个体fitness函数"""
    # 这使用的是rank_ic
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


def fitness_population(df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
    """种群fitness函数"""
    if df is None:
        return {}, {}, {}, {}

    df = df.group_by('date').agg(
        [fitness_individual(X, label) for X in columns]
    ).sort(by=['date']).fill_nan(None)
    # 将IC划分成训练集与测试集
    df_train = df.filter(pl.col('date') < split_date)
    df_valid = df.filter(pl.col('date') >= split_date)

    ic_train = df_train.select(cs.numeric().mean())
    ir_train = df_train.select(cs.numeric().mean() / cs.numeric().std(ddof=0))
    ic_valid = df_valid.select(cs.numeric().mean())
    ir_valid = df_valid.select(cs.numeric().mean() / cs.numeric().std(ddof=0))

    ic_train = ic_train.to_dicts()[0]
    ir_train = ir_train.to_dicts()[0]
    ic_valid = ic_valid.to_dicts()[0]
    ir_valid = ir_valid.to_dicts()[0]

    return ic_train, ir_train, ic_valid, ir_valid


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
    ic_train, ir_train, ic_valid, ir_valid = fitness_population(df_output, list(exprs_dict.keys()), label=label, split_date=split_date)
    logger.info("{}代{}批 适应度 计算完成", gen, batch_id)

    # 样本内外适应度提取
    new_results = {}
    for k, v in exprs_dict.items():
        v = str(v)
        new_results[v] = {'ic_train': get_fitness(k, ic_train),
                          'ir_train': get_fitness(k, ir_train),
                          'ic_valid': get_fitness(k, ic_valid),
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
            s0, s1, s2, s3 = d['ic_train'], d['ir_train'], d['ic_valid'], d['ir_valid']
            # ic要看绝对值
            s0, s1, s2, s3 = abs(s0), s1, abs(s2), s3
            # TODO 这地方要按自己需求定制，过滤太多可能无法输出有效表达式
            if s0 == s0:  # 非空
                if s0 > 0.001:  # 样本内打分要大
                    if s0 * 0.6 < s2:  # 样本外打分大于样本内打分的70%
                        results.append((s0, s2))
                        continue
        results.append((np.nan, np.nan))

    return results
