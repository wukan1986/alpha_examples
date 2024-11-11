import copy
from typing import Dict

from expr_codegen.codes import sources_to_exprs
from expr_codegen.expr import is_meaningless
from loguru import logger
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
    if isinstance(e, (int, float)):
        return True
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
    # sources.insert(0, 'GP_000=1') # DEBUG
    raw, exprs_dict = sources_to_exprs(globals_, '\n'.join(sources), convert_xor=False)
    return exprs_dict


def filter_exprs(exprs_dict, pset, RET_TYPE, fitness_results):
    before_len = len(exprs_dict)
    # 清理重复表达式，通过字典特性删除
    exprs_dict = {v: k for k, v in exprs_dict.items()}
    exprs_dict = {v: k for k, v in exprs_dict.items()}
    # 清理非法表达式
    exprs_dict = {k: v for k, v in exprs_dict.items() if not is_invalid(v, pset, RET_TYPE)}
    # 清理无意义表达式
    exprs_dict = {k: v for k, v in exprs_dict.items() if not is_meaningless(v)}
    after_len = len(exprs_dict)
    logger.info('剔除重复、非法、无意义表达式，数量由 {} -> {}', before_len, after_len)

    # 历史表达式不再重复计算
    before_len = len(exprs_dict)
    exprs_dict = {k: v for k, v in exprs_dict.items() if str(v) not in fitness_results}
    after_len = len(exprs_dict)
    logger.info('剔除历史已经计算过适应度的表达式，数量由 {} -> {}', before_len, after_len)
    return exprs_dict
