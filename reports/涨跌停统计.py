"""
1. 每天的涨跌停比
2.

"""
import pathlib

import polars as pl
from expr_codegen import codegen_exec

风格板块 = [
    'AB股',
    'AH股',
    'B股',
    '百元股',
    '标准普尔',
    '参股保险',
    '参股期货',
    '参股券商',
    '参股新三板',
    '参股银行',
    '创业板综',
    '创业成份',
    '低价股',
    '富时罗素',
    'GDR',
    '股权激励',
    '股权转让',
    '沪股通',
    'HS300_',
    '机构重仓',
    '基金重仓',
    '举牌',
    'MSCI中国',
    '茅指数',
    '内贸流通',
    '宁组合',
    '破净股',
    'QFII重仓',
    # '融资融券',
    '社保重仓',
    '深成500',
    '深股通',
    'ST股',
    '深证100R',
    '上证180_',
    '上证380',
    '上证50_',
    '预亏预减',
    '养老金',
    '央视50_',

    '预盈预增',
    '证金持股',
    '专精特新',
    '昨日触板',
    '昨日连板',
    '昨日连板_含一字',
    '昨日涨停',
    '昨日涨停_含一字',
    '中特估',
    '中证500',
    # '转债标的',
    '中字头',
    # '国企改革',
]

地区板块 = [
    '长江三角',
    '西部大开发',
]

# 建议每个月更新概念板块一次
files = list(pathlib.Path(r"D:\data\akshare\stock_board_concept_cons_em\20241204").glob(f'*.parquet'))
df_board = []
for file in files:
    # 部分概念已经过时，剔除
    if file.stem in 风格板块:
        continue
    if file.stem in 地区板块:
        continue
    # 停牌股票成交量nan，导致多文件int与float格式冲突，这里强制转float
    df_board.append(pl.read_parquet(file).with_columns(pl.col("成交量").cast(pl.Float64)))
df_board = pl.concat(df_board)
print(df_board.schema)
df_board = df_board.select('代码', '名称', 'board')

# 读取个股数据
df_stock = pl.read_parquet(r"M:\preprocessing\data2.parquet")
# 读取分时数据
df_min = pl.read_parquet(r"分时炸板.parquet")
df_stock = df_stock.join(df_min, how='left', on=['asset', 'date'])
df_stock = df_stock.with_columns(code=pl.col('asset').str.slice(0, 6))
print(df_stock.schema)


def _code_block_1():
    # 注意：由于没有买卖一价，这里的涨停只是成交价触及涨停价，卖一价可能还有挂单
    上涨 = CLOSE > CLOSE[1]
    阳线 = close > open
    收盘涨停 = close >= high_limit - 0.001
    最高涨停 = high >= high_limit - 0.001
    最低涨停 = low >= high_limit - 0.001
    收盘跌停 = close <= low_limit + 0.001
    连板天数 = ts_cum_sum_reset(收盘涨停)
    涨停T天, 涨停N板, _ = ts_up_stat(收盘涨停)
    放量 = volume / ts_mean(volume, 3)[1]
    炸板 = 最高涨停 & ~收盘涨停
    晋级 = 连板天数 > 1
    首板 = 连板天数 == 1
    一进二 = 连板天数 == 2
    二进三 = 连板天数 == 3
    昨日涨停 = 收盘涨停[1]
    昨日炸板 = 炸板[1]
    收益率_CC = CLOSE / CLOSE[1] - 1  # 昨天排板入场，今天收盘出场
    收益率_CO = OPEN / CLOSE[1] - 1  # 昨天排板入场，今天开盘出场


df_stock = codegen_exec(df_stock, _code_block_1, over_null="partition_by")
# 观察本年
DATE1 = pl.date(2024, 1, 1)
df_stock = df_stock.filter(pl.col('date') >= DATE1)
# 可以加入板块，热门板块晋级率更高
df_stock.group_by('date').agg(
    上涨家数=pl.col('上涨').sum(),
    阳线家数=pl.col('阳线').sum(),
    涨停家数=pl.col('收盘涨停').sum(),
    跌停家数=pl.col('收盘跌停').sum(),
    高停家数=pl.col('最高涨停').sum(),
    一字家数=pl.col('最低涨停').sum(),
    炸板家数=pl.col('炸板').sum(),
    连板家数=pl.col('晋级').sum(),
    连板高度=pl.col('连板天数').max(),
    晋级率=pl.col('晋级').sum() / pl.col('昨日涨停').sum(),
    昨日涨停表现=pl.col('收益率_CC').filter(pl.col('昨日涨停')).mean(),
    昨日炸板表现=pl.col('收益率_CC').filter(pl.col('昨日炸板')).mean(),
    昨日涨停高开=pl.col('收益率_CO').filter(pl.col('昨日涨停')).mean(),
).with_columns(
    炸板率=pl.col('炸板家数') / pl.col('高停家数'),
).sort('date', descending=True).write_csv('涨跌停统计.csv')
# 如果涨停指数长期维持在两个点以内，当日涨停数和炸板数维持2比1以内，炸板指数跌幅1.5开外，那定义为弱势

# 观察某几天
DATE2 = pl.date(2024, 12, 23)
df_stock = df_stock.filter(pl.col('date') >= DATE2)

df_stock_board = df_stock.join(df_board, how='left', left_on=['code'],
                               right_on=['代码'])  # .filter(pl.col('board').is_not_null())
del df_stock
del df_board

columns = ['名称', "涨停T天", "涨停N板", "收盘涨停", '放量', '炸板次数', '首次封板', '最后封板']

df_board_sorted = df_stock_board.group_by('date', 'board').agg(
    pl.col('收盘涨停').sum().alias('封板数'),
    (pl.col('连板天数') > 1).sum().alias('连板数'),
    pl.col("asset", *columns).top_k_by('涨停N板', 5),
).sort('date', '封板数', descending=[False, True]).with_row_index()
df_board_sorted.explode("asset", *columns).write_csv('涨停板块前5.csv')
print(df_board_sorted)

df_stock_board = df_stock_board.join(df_board_sorted.select('index', 'date', 'board'), how='left', on=['date', 'board'])
df_stock_board = df_stock_board.sort('date', 'asset', 'index')

df_stock_list = df_stock_board.group_by('date', 'asset').agg(pl.col(*columns).first(), pl.col('board').str.join(','))
df_stock_list.sort('date', '涨停N板', descending=[False, True]).write_csv('涨停个股.csv')
print(df_stock_list)
