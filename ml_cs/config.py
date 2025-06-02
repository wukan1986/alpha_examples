"""
数据中会有空值，LightGBM等库支持nan,但sklearn不支持nan
标签在训练时时不能有空值，在预测时不关心
特征不能有空值
"""
import polars as pl  # noqa
import polars.selectors as cs  # noqa
from polars_ta.prefix.wq import cs_zscore

# %%
DATE = "date"
ASSET = "asset"
LABEL = 'LABEL'  # 训练用的标签
FWD_RET = 'FWD_RET'  # 计算净值必需提供1日收益率
DATA_END = '2025-03'
DATA_START = '2025-04'

INPUT1_PATH = r'M:\preprocessing\out1.parquet'  # 添加了特征的数据

# %%
MODEL_FILENAME = r'D:\GitHub\alpha_examples\ml_cs\models.pkl'  # 训练后保存的模型名
PRED_PATH = 'pred.parquet'  # 预测结果
PRED_EXCEL = 'pred.xlsx'  # 预测结果导出Excel

# %%
# TODO 丢弃的字段。保留的字段远远多余丢弃的字段，用丢弃法
# 1. 对机器学习无意义的字段
# 2. 留下日期、资产、多个特征、一标签、一未来收益
drop_columns = [
    'paused', 'factor',
    'high_limit', 'low_limit',
    'sw_l1', 'sw_l3', 'sw_l2', 'zjw',
    '上海主板', '深圳主板', '科创板', '创业板', '北交所',
    'NEXT_DOJI4',
    'SSE50', 'CSI300', 'CSI500', 'CSI1000',
    'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'pe_ratio_lyr',
    "ONE", "MC_LOG", "MC_NORM", 'market_cap', 'circulating_market_cap',
]

# TODO 分类特征。布尔型号和少量的整数型，只在LightGBM中使用
# 为何只在训练时使用？预测时不需要吗？
categorical_feature = [
    'DOJI4',
    # '短期上穿中期',
    # '当前价格是否高于10日均线',
]

exclude_columns = [
]


# %%
def load_process():
    """加载数据，然后进行预处理"""
    df = pl.read_parquet(INPUT1_PATH)
    print(df.columns)

    # 删除不需要的字段。留下日期、资产、多个特征、一标签、一未来收益
    df = df.drop(drop_columns)

    # 预处理
    df = df.with_columns(
        cs_zscore(cs.float() & cs.exclude(DATE, ASSET, LABEL, FWD_RET, *exclude_columns)).over(DATE)
    )
    return df
