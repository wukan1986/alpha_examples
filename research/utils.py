import polars as pl


def with_industry(df: pl.DataFrame, industry_name: str):
    """添加行业列"""
    df = df.with_columns([
        # 行业处理，由浮点改成整数
        pl.col(industry_name).cast(pl.UInt32),
    ])
    # TODO 没有行业的也过滤，这会不会有问题？
    df = df.filter(
        #
        pl.col(industry_name).is_not_null(),
    )
    # TODO drop_first丢弃哪个字段是随机的，非常不友好，只能在行业中性化时动态修改代码
    df = df.with_columns(df.to_dummies(industry_name, drop_first=True))
    # industry_columns = list(filter(lambda x: re.search(rf"^{industry_name}_\d+$", x), df.columns))
    return df
