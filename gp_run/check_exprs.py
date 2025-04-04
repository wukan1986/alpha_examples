# !!! 以下代码在VSCode或Notebook中执行更好，能显示LATEX表达式和画表达式树状图
# %%

# %%
# 从main中导入，可以大大减少代码
from gp_run.main import *

with open(f'log/exprs_0001.pkl', 'rb') as f:
    pop = pickle.load(f)

print_population(pop, globals().copy())
