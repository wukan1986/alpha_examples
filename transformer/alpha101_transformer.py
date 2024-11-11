import ast
import re

from expr_codegen.codes import source_replace, RenameTransformer, SyntaxTransformer

encoding = 'utf-8'
input_file = 'alpha101.txt'
output_file = 'alpha101_out.txt'


def code_replace(source):
    source = re.sub('Alpha#(.+?):', lambda m: f'alpha_{m.group(1).zfill(3)}=', source)
    source = source.replace('||', '|').replace('IndClass.', '')
    return source


# ==========================
# 观察区，查看存在哪些变量和函数
with open(input_file, 'r', encoding=encoding) as f:
    sources = f.readlines()

    # 不要太大，防止内存不足
    source = '\n'.join(sources[:1000])
    source = code_replace(source)
    tree = ast.parse(source_replace(source))
    st = RenameTransformer({}, {})
    st.visit(tree)

    print('=' * 60)
    print(st.funcs_old)
    print(st.args_old)
    print(st.targets_old)
    print('=' * 60)
    # print(ast.unparse(tree))

# ==========================
# 映射
funcs_map = {'SignedPower': 'signed_power',
             'rank': 'cs_rank',
             'delta': 'ts_delta',
             'delay': 'ts_delay',
             'IndNeutralize': 'gp_demean',
             'correlation': 'ts_corr',
             'Sign': 'sign',
             'ts_argmax': 'ts_arg_max',
             'max': 'max_',
             'Ts_ArgMin': 'ts_arg_min',
             'indneutralize': 'gp_demean',
             'scale': 'cs_scale',
             'Ts_Rank': 'ts_rank',
             'sum': 'ts_sum',
             'ts_argmin': 'ts_arg_min',
             'Ts_ArgMax': 'ts_arg_max',
             'Log': 'log',
             'covariance': 'ts_covariance',
             'stddev': 'ts_std_dev',
             'decay_linear': 'ts_decay_linear',
             'abs': 'abs_',
             'product': 'ts_product',
             'min': 'min_',
             }
args_map = {}
targets_map = {}


# TODO 如果后面文件太大，耗时太久，需要手工放开后面一段
# sys.exit(-1)
# ==========================
class Alpha101Transformer(ast.NodeTransformer):
    def visit_Call(self, node):
        # 部分函数需要
        if node.func.id in ('ts_sum', 'ts_corr', 'ts_delta', 'ts_arg_max',
                            'ts_arg_min', 'ts_std_dev', 'ts_decay_linear',
                            'ts_product', 'ts_rank', 'ts_min', 'ts_max', 'ts_covariance'):
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    arg.value = round(arg.value)

        if node.func.id == 'gp_demean':
            node.args = list(reversed(node.args))

        for arg in node.args:
            if isinstance(arg, ast.Name):
                arg.id = arg.id.upper()

        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        if isinstance(node.left, ast.Name):
            node.left.id = node.left.id.upper()
        if isinstance(node.right, ast.Name):
            node.right.id = node.right.id.upper()

        self.generic_visit(node)
        return node

    def visit_Compare(self, node):
        if isinstance(node.left, ast.Name):
            node.left.id = node.left.id.upper()
        for com in node.comparators:
            if isinstance(com, ast.Name):
                com.id = com.id.upper()

        self.generic_visit(node)
        return node


with open(input_file, 'r', encoding=encoding) as f:
    sources = f.readlines()

    t1 = SyntaxTransformer()
    t1.convert_xor = True
    st = RenameTransformer(funcs_map, args_map, targets_map)

    at = Alpha101Transformer()

    outputs = []
    for i in range(0, len(sources), 1000):
        print(i)
        source = '\n'.join(sources[i:i + 1000])
        source = code_replace(source)

        tree = ast.parse(source_replace(source))
        t1.visit(tree)
        st.visit(tree)
        at.visit(tree)
        outputs.append(ast.unparse(tree))

    print('转码完成')
    with open(output_file, 'w') as f2:
        f2.writelines(outputs)
    print('保存成功')
