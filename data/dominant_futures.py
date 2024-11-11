"""
聚宽999主力合约只是简单将主力拼接起来，换月时会有跳空问题，计算收益时错误

这里提供示例生成后复权因子和对应的价格

分为乘除复权和加减复权，
加减复权适合用手数记录持仓的情况
乘除复权适合用于资金百分比等资产配置

"""
from typing import List

import polars as pl
import polars.selectors as cs
from expr_codegen.tool import codegen_exec
from polars_ta.wq import ts_delay, ts_cum_prod


def filter_assets(df: pl.DataFrame, assets: List[str], exclude: bool = True, asset='asset') -> pl.DataFrame:
    # 使用product时防止B与BB被starts_with匹配了
    exprs = [pl.col(asset) == a for a in assets]
    # exprs = [pl.col(asset).str.starts_with(a) for a in assets]
    expr = pl.any_horizontal(exprs)
    if exclude:
        expr = ~expr

    df = df.filter(expr)
    return df


# 加载全部行情
df1 = pl.read_parquet(r'M:\data\jqresearch\get_price_futures_daily', use_pyarrow=True).rename({'time': 'date', 'code': 'asset'})

# 加载换月信息，其中product是品种，asset是合约
df2 = pl.read_parquet(r'M:\data\jqresearch\get_dominant_futures', use_pyarrow=True).rename({'__index_level_0__': 'date'})
df2 = df2.unpivot(index="date", on=cs.numeric(), value_name='asset', variable_name='product')
# join时时间格式要统一
df2 = df2.with_columns(pl.col('date').str.to_datetime(time_unit='us'))

# 提前排除流动性不好的品种
drop_assets = ['WH', 'RI', 'PM', 'ZC', 'LR', 'JR', 'RS', 'BB', 'WR', 'RR', 'CY', 'FB', 'BC']
df2 = filter_assets(df2, drop_assets, exclude=True, asset='product')


def _code_block_1():
    # 同asset下的移动
    close_1 = ts_delay(close, 1)


def _code_block_mul_div():
    # 同product的移动
    # 乘除后复权因子
    factor = ts_cum_prod(ts_delay(close, 1) / close_1)
    # 后复权开高低收
    OPEN = open * factor
    HIGH = high * factor
    LOW = low * factor
    CLOSE = close * factor


def _code_block_add_sub():
    # 同product的移动
    # 加减后复权因子
    factor = ts_cum_sum(ts_delay(close, 1) - close_1)
    # 后复权开高低收
    OPEN = open + factor
    HIGH = high + factor
    LOW = low + factor
    CLOSE = close + factor


df1 = codegen_exec(df1, _code_block_1)
df3 = df2.join(df1, on=['date', 'asset'], how='left', coalesce=True)
del df1
del df2
# TODO !!! 非常重要，分组是用product而不是asset，否则结果是错的
df3 = codegen_exec(df3, _code_block_mul_div, asset='product')
print(df3)
df4 = df3.filter(pl.col('product') == 'ZN')
print(df4.to_pandas())
