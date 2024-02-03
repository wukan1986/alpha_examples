"""
1. 准备数据
    date,asset,features,..., returns,...,labels

features建议提前做好预处理。因为在GP中计算效率低下，特别是行业中性化等操作强烈建议在提前做。因为
1. `ts_`。按5000次股票，要计算5000次
2. `cs_`。按1年250天算，要计算250次
3. `gp_`计算次数是`cs_`计算的n倍。按30个行业，1年250天，要计算30*250=7500次

ROCP=ts_return，不移动位置，用来做特征。前移shift(-x)就只能做标签了

returns是shift前移的简单收益率，用于事后求分组收益
1. 对数收益率方便进行时序上的累加
2. 简单收益率方便横截面上进行等权
log_return = ln(1+simple_return)

labels是因变量
1. 可能等于returns
2. 可能是超额收益率
3. 可能是0/1等分类标签

样本内外的思考
一开始，为了计算样本内外，我参考了机器学习的方法，提前将数据分成了训练集和测试集，然后分别计算因子值和IC/IR，
运行了一段时间后忽然发现，对于个体来说表达式已经明确了，每日的IC已经是确定了，而训练集与测试集的IC区别只是mean(IC)时间段的差异。
所以整体一起计算，然后分段计算IC/IR速度能快上不少。

"""
import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import operator
import pickle
import time
from datetime import datetime
from itertools import count
from typing import Sequence, Dict

import polars as pl
import polars.selectors as cs
from deap import base, creator
from expr_codegen.codes import sources_to_exprs
from expr_codegen.expr import is_meaningless
from expr_codegen.tool import ExprTool
from loguru import logger
from more_itertools import batched

from gp.custom import add_constants, add_operators, add_factors, RET_TYPE
# !!! 非常重要。给deap打补丁
from gp.deap_patch import *  # noqa
from gp.helper import stringify_for_sympy, is_invalid
# 引入OPEN等
from sympy_define import *  # noqa

logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题
logger.add(sys.stderr, level='INFO')  # 只输出INFO以上的日志
# ======================================
# TODO 必须元组，1表示找最大值,-1表示找最小值
FITNESS_WEIGHTS = (1.0, 1.0)

# TODO y表示类别标签、因变量、输出变量，需要与数据文件字段对应
LABEL_y = 'RETURN_OO_1'
# TODO 种群如果非常大，但内存比较小，可以分批计算，每次计算BATCH_SIZE个个体
BATCH_SIZE = 50

# TODO: 数据准备，脚本将取df_input，可运行`data`下脚本生成
df_input = pl.read_parquet('data/data.parquet')
dt1 = datetime(2021, 1, 1)
# ======================================
# 日志路径
LOG_DIR = Path('log')
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================
def fitness_individual(a: str, b: str) -> pl.Expr:
    """个体fitness函数"""
    # 这使用的是rank_ic
    return pl.corr(a, b, method='spearman', ddof=0, propagate_nans=False)


def fitness_population(df: pl.DataFrame, columns: Sequence[str], label: str, split_date: datetime):
    """种群fitness函数"""
    if df is None:
        return {}, {}, {}, {}

    df = df.group_by(by=['date']).agg(
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


def get_fitness(name: str, kv: Dict[str, float]) -> float:
    return kv.get(name, False) or float('nan')


def population_to_exprs(pop):
    """群体转表达式"""
    sources = [f'GP_{i:04d}={stringify_for_sympy(expr)}' for i, expr in enumerate(pop)]
    raw, exprs_dict = sources_to_exprs(globals().copy(), '\n'.join(sources))
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


def batched_exprs(exprs_dict, gen, batch_id, label, df_input, split_date):
    """每代种群分批计算

    由于种群数大，一次性计算可能内存不足，所以提供分批计算功能，同时也为分布式计算做准备
    """
    tool = ExprTool()
    # 表达式转脚本
    codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                        replace=False, regroup=True, format=True,
                        date='date', asset='asset')

    # 备份生成的代码
    path = LOG_DIR / f'codes_{gen:04d}_{batch_id:02d}.py'
    import_path = f'log.codes_{gen:04d}_{batch_id:02d}'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(codes)

    cnt = len(exprs_dict)
    logger.info("{}代{}批 代码 开始执行。共 {} 条 表达式", gen, batch_id, cnt)
    tic = time.perf_counter()

    # exec和import都可以，import好处是内部代码可调试
    _lib = __import__(import_path, fromlist=['*'])

    # 因子计算
    df_output = _lib.main(df_input)

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


def map_exprs(evaluate, invalid_ind, gen, label, df_input, split_date, batch_size):
    """原本是一个普通的map或多进程map，个体都是独立计算
    但这里考虑到表达式很相似，可以重复利用公共子表达式，
    所以决定种群一起进行计算，返回结果评估即可
    """
    g = next(gen)
    # 保存原始表达式，立即保存是防止崩溃后丢失信息, 注意：这里没有存fitness
    with open(LOG_DIR / f'exprs_{g:04d}.pkl', 'wb') as f:
        pickle.dump(invalid_ind, f)

    # 读取历史fitness
    try:
        with open(LOG_DIR / f'fitness_cache.pkl', 'rb') as f:
            fitness_results = pickle.load(f)
    except FileNotFoundError:
        fitness_results = {}

    logger.info("表达式转码...")
    # DEAP表达式转sympy表达式。约定以GP_开头，表示遗传编程
    exprs_dict = population_to_exprs(invalid_ind)
    exprs_old = exprs_dict.copy()
    exprs_dict = filter_exprs(exprs_dict, pset, RET_TYPE, fitness_results)

    if len(exprs_dict) > 0:
        # 分批计算,以后可以考虑改成并行
        for batch_id, exprs_batched in enumerate(batched(exprs_dict.items(), batch_size)):
            new_results = batched_exprs(dict(exprs_batched), g, batch_id, label, df_input, split_date)

            # 合并历史与最新的fitness
            fitness_results.update(new_results)
            # 保存适应度，方便下一代使用
            with open(LOG_DIR / f'fitness_cache.pkl', 'wb') as f:
                pickle.dump(fitness_results, f)
    else:
        pass

    # 取评估函数值，多目标。
    results2 = []
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
                        results2.append((s0, s2))
                        continue
        results2.append((np.nan, np.nan))

    return results2


# ======================================
# 这里的ret_type只要与addPrimitive对应即可
pset = gp.PrimitiveSetTyped("MAIN", [], RET_TYPE)
pset = add_constants(pset)
pset = add_operators(pset)
pset = add_factors(pset)

# 可支持多目标优化
creator.create("FitnessMulti", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize=3)  # 目标优化
# toolbox.register("select", tools.selNSGA2)  # 多目标优化 FITNESS_WEIGHTS = (1.0, 1.0)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("evaluate", print)  # 不单独做评估了，在map中一并做了
toolbox.register('map', map_exprs, gen=count(), label=LABEL_y, df_input=df_input, split_date=dt1, batch_size=BATCH_SIZE)


def main():
    # TODO: 伪随机种子，同种子可复现
    random.seed(9527)

    # TODO: 初始种群大小
    pop = toolbox.population(n=100)
    # TODO: 名人堂，表示最终选优多少个体
    hof = tools.HallOfFame(100)

    # 只统计一个指标更清晰
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # 打补丁后，名人堂可以用nan了，如果全nan会报警
    stats.register("avg", np.nanmean, axis=0)
    stats.register("std", np.nanstd, axis=0)
    stats.register("min", np.nanmin, axis=0)
    stats.register("max", np.nanmax, axis=0)

    # 使用修改版的eaMuPlusLambda
    population, logbook = eaMuPlusLambda(pop, toolbox,
                                         # 选多少个做为下一代，每次生成多少新个体
                                         mu=150, lambda_=100,
                                         # 交叉率、变异率，代数
                                         cxpb=0.5, mutpb=0.1, ngen=2,
                                         # 名人堂参数
                                         # alpha=0.05, beta=10, gamma=0.25, rho=0.9,
                                         stats=stats, halloffame=hof, verbose=True,
                                         # 早停
                                         early_stopping_rounds=2)

    return population, logbook, hof


def print_population(pop):
    # !!!这句非常重要
    exprs_dict = population_to_exprs(pop)
    for (k, v), i in zip(exprs_dict.items(), pop):
        print(f'{k}', '\t', i.fitness, '\t', v, '\t<--->\t', i)


if __name__ == "__main__":
    print('另行执行`tensorboard --logdir=runs`，然后在浏览器中访问`http://localhost:6006/`，可跟踪运行情况')
    logger.warning('运行前请检查`fitness_cache.pkl`是否要手工删除。数据集、切分时间发生了变化一定要删除，否则重复的表达式不会参与计算')

    population, logbook, hof = main()

    # 保存名人堂
    with open(LOG_DIR / f'hall_of_fame.pkl', 'wb') as f:
        pickle.dump(hof, f)

    print('=' * 60)
    print(logbook)

    print('=' * 60)
    print_population(hof)
