import polars as pl
from expr_codegen.tool import codegen_exec

# 从日线中取涨跌停
df_1d = (pl.read_parquet(r"M:\preprocessing\data1.parquet")
         .rename({"time": "date", "code": "asset"})
         .select("date", "asset", "high_limit", "low_limit"))

# 加载分钟数据
df_1m = (pl.read_parquet(r"D:\data\jqresearch\get_price_stock_minute\20241*.parquet")
         .rename({"time": "datetime", "code": "asset"})
         .select("datetime", "asset", "open", "high", "close", "paused")
         .with_columns(pl.col("datetime").cast(pl.Datetime('us'))))
df_1m = df_1m.filter(pl.col("paused") == 0)
df_1m = df_1m.with_columns(date=pl.col("datetime").dt.truncate('1d'))

df = df_1m.join(df_1d, on=["date", "asset"])
del df_1m
del df_1d


def _code_block_1():
    # 一下表达式都是用于分钟。因为没有tick数据只能用分钟模拟
    开盘涨停 = open >= high_limit - 0.001
    最高涨停 = high >= high_limit - 0.001
    收盘涨停 = close >= high_limit - 0.001
    昨收涨停 = ts_delay(收盘涨停, 1, False)
    # 化简后逻辑表达式更复杂了
    封板 = (~昨收涨停 & 开盘涨停) | (~开盘涨停 & 最高涨停)
    炸板 = (昨收涨停 & ~开盘涨停) | (最高涨停 & ~收盘涨停)


df = df.with_columns(_asset_date=pl.struct("asset", "date"))
df = codegen_exec(df, _code_block_1, asset="_asset_date", output_file="1_out.py")

df = df.group_by("asset", "date").agg(
    炸板次数=pl.col("炸板").sum(),
    首次封板=pl.col("datetime").filter(pl.col('封板')).first(),
    最后封板=pl.col("datetime").filter(pl.col('封板')).last(),
)

df = df.filter(pl.col("炸板次数") >= 5).sort("asset", "date")

#
print(df)
#
