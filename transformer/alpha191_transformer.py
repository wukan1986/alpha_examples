import ast
import re

from expr_codegen.codes import source_replace, RenameTransformer, SyntaxTransformer

encoding = 'utf-8'
input_file = 'alpha191.txt'
output_file = 'alpha191_out.txt'


def code_replace(source):
    source = re.sub('Alpha(.+?) ', lambda m: f'alpha_{m.group(1).zfill(3)}=', source)
    source = source.replace('||', '|')
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
funcs_map = {'TSMIN': 'ts_min',
             'SUM': 'ts_sum',
             'MEAN': 'ts_mean',
             'SIGN': 'sign',
             'FILTER': '$',
             'SUMIF': '$',
             'LOWDAY': 'ts_arg_min',
             'MIN': 'min_',
             'DELAY': 'ts_delay',
             'STD': 'ts_std_dev',
             'CORR': 'ts_corr',
             'WMA': 'ts_WMA',  # ts_decay_linear?
             'PROD': 'ts_product',
             'SEQUENCE': '$',
             'RANK': 'cs_rank',
             'TSRANK': 'ts_rank',
             'LOG': 'log',
             'TSMAX': 'ts_max',
             'COUNT': 'ts_count',
             'REGBETA': '$',
             'MA': 'ts_mean',
             'MAX': 'max_',
             'DECAYLINEAR': 'ts_decay_linear',
             'COVIANCE': 'ts_covariance',
             'REGRESI': '$',
             'DELTA': 'ts_delta',
             'HIGHDAY': 'ts_arg_max',
             'SUMAC': '$',
             'ABS': 'abs_',
             'SMA': 'ts_SMA_CN',
             }
args_map = {'HGIH': 'HIGH'}
targets_map = {}


# TODO 如果后面文件太大，耗时太久，需要手工放开后面一段
# sys.exit(-1)


# ==========================
class Alpha191Transformer(ast.NodeTransformer):

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

    t1 = SyntaxTransformer(True)
    st = RenameTransformer(funcs_map, args_map, targets_map)

    at = Alpha191Transformer()

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
