"""
数据中会有空值，LightGBM等库支持nan,但sklearn不支持nan
标签在训练时时不能有空值，在预测时不关心
特征不能有空值
"""
import polars as pl  # noqa
import polars.selectors as cs  # noqa

# %%
DATE = "date"
ASSET = "asset"
LABEL = 'LABEL'  # 训练用的标签
FWD_RET = 'FWD_RET'  # 计算净值必需提供日化收益率
DATA_END = '2025-03'
DATA_START = '2025-04'

INPUT1_PATH = r'M:\preprocessing\data5.parquet'  # 添加了特征的数据

# %%
MODEL_FILENAME = r'D:\GitHub\alpha_examples\ml_cs\models.pkl'  # 训练后保存的模型名
PRED_PATH = 'pred.parquet'  # 预测结果
PRED_EXCEL = 'pred.xlsx'  # 预测结果导出Excel

# %%
# TODO 特征
feature_columns = [
    "MC_NEUT", "EP", "BP", "SP", "CFP",

    "DOJI4",

    "A_0001", "A_0002", "A_0003",
]

# TODO 分类特征。布尔型号和少量的整数型，只在LightGBM中使用
# 为何只在训练时使用？预测时不需要吗？
categorical_feature = [
    'DOJI4',
    # '短期上穿中期',
    # '当前价格是否高于10日均线',
]


# %%
def load_process():
    """加载数据，然后进行预处理"""
    df: pl.DataFrame = pl.read_parquet(INPUT1_PATH)
    print(df.columns)

    # 留下日期、资产、多个特征、一标签、一未来收益
    df = df.select(DATE, ASSET, LABEL, FWD_RET, *feature_columns)

    # 预处理，需要提前在其他地方处理好，这里不再处理
    # df = df.with_columns(
    #     cs_zscore(cs.float() & cs.exclude(DATE, ASSET, LABEL, FWD_RET, *exclude_columns)).over(DATE)
    # )

    # TODO 回归问题转换成分类问题
    df = df.with_columns(
        (pl.col(LABEL) > 0.00).cast(pl.UInt8)
    )
    print(df[LABEL].value_counts())
    return df
