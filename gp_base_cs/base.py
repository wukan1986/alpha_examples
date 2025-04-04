import copy
from typing import Dict

from expr_codegen.codes import sources_to_exprs
from expr_codegen.expr import is_meaningless
from loguru import logger
from sympy import Basic, Function, symbols, preorder_traversal


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)

    converter = {
        'aa_sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'aa_div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'aa_mul': lambda *args_: "Mul({},{})".format(*args_),
        'aa_add': lambda *args_: "Add({},{})".format(*args_),
        'aa_max': lambda *args_: "max_({},{})".format(*args_),
        'aa_min': lambda *args_: "min_({},{})".format(*args_),

        'ai_sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'ai_div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'ai_mul': lambda *args_: "Mul({},{})".format(*args_),
        'ai_add': lambda *args_: "Add({},{})".format(*args_),
        'ai_max': lambda *args_: "max_({},{})".format(*args_),
        'ai_min': lambda *args_: "min_({},{})".format(*args_),

        'ia_sub': lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        'ia_div': lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        'ia_mul': lambda *args_: "Mul({},{})".format(*args_),
        'ia_add': lambda *args_: "Add({},{})".format(*args_),
        'ia_max': lambda *args_: "max_({},{})".format(*args_),
        'ia_min': lambda *args_: "min_({},{})".format(*args_),
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


def get_node_name(node):
    """得到节点名"""
    if hasattr(node, 'name'):
        # 如 ts_arg_max
        node_name = node.name
    else:
        # 如 Add
        node_name = str(node.func.__name__)
    return node_name


def convert_inverse_sympy(e):
    if not isinstance(e, Basic):
        return e

    replacements = []
    for node in preorder_traversal(e):
        node_name = get_node_name(node)

        if node_name in ('max_', 'min_'):
            func_name = list(f"aa_{node_name}")[:-1]
            if node.args[0].is_Number:
                func_name[0] = 'i'
            if node.args[1].is_Number:
                func_name[1] = 'i'
            func = symbols(''.join(func_name), cls=Function)
            replacements.append((node, func(node.args[0], node.args[1])))

        if node_name == 'Add':
            last_node = node.args[0]
            for arg2 in node.args[1:]:
                func_name = list('aa_add')
                if last_node.is_Number:
                    func_name[0] = 'i'
                if arg2.is_Number:
                    func_name[1] = 'i'
                func = symbols(''.join(func_name), cls=Function)
                last_node = func(last_node, arg2)
            replacements.append((node, last_node))

        if node_name == 'Mul':
            last_node = node.args[0]
            for arg2 in node.args[1:]:
                func_name = list('aa_mul')
                if last_node.is_Number:
                    func_name[0] = 'i'
                if arg2.is_Number:
                    func_name[1] = 'i'
                func = symbols(''.join(func_name), cls=Function)
                last_node = func(last_node, arg2)
            replacements.append((node, last_node))

    for node, replacement in replacements:
        print(node, '  ->  ', replacement)
        e = e.xreplace({node: replacement})
    return e


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


def print_population(population, globals_, more=True):
    """打印种群"""
    exprs_dict = population_to_exprs(population, globals_)

    if more:
        # 打印时更好看
        for (k, v), i in zip(exprs_dict.items(), population):
            print(f'{k}', '\t', i.fitness, '\t', v, '\t<--->\t', i)
    else:
        # 输出到expr_codegen时更方便
        for (k, v), i in zip(exprs_dict.items(), population):
            print(f'{k}={v}')


def population_to_exprs(population, globals_):
    """群体转表达式"""
    if len(population) == 0:
        return {}
    sources = [f'GP_{i:04d}={stringify_for_sympy(expr)}' for i, expr in enumerate(population)]
    # sources.insert(0, 'GP_000=1') # DEBUG
    raw, exprs_dict = sources_to_exprs(globals_, '\n'.join(sources), convert_xor=False)
    return exprs_dict


def strings_to_sympy(population, globals_):
    """字符串转表达式"""
    if len(population) == 0:
        return {}
    sources = [f'GP_{i:04d}={expr}' for i, expr in enumerate(population)]
    raw, exprs_dict = sources_to_exprs(globals_, '\n'.join(sources), convert_xor=False)
    exprs_dict = {k: convert_inverse_sympy(v) for k, v in exprs_dict.items()}
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
