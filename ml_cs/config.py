"""
数据中会有空值，LightGBM等库支持nan,但sklearn不支持nan
标签在训练时时不能有空值，在预测时不关心
特征不能有空值
"""

# %%
DATE = "date"
ASSET = "asset"
MODEL_FILENAME = r'D:\GitHub\alpha_examples\ml_cs\models.pkl'  # 训练后保存的模型名
INPUT1_PATH = r'M:\preprocessing\out1.parquet'  # 添加了特征的数据
LABEL = 'LABEL'  # 训练用的标签
FWD_RET = 'RET'  # 计算净值必需提供1日收益率
DATA_END = '2025-03'
DATA_START = '2025-04'

# TODO 丢弃的字段。保留的字段远远多余丢弃的字段，用丢弃法
# 1. 对机器学习无意义的字段
# 2. 多余的未来数据.仅保留一标签一未来收益
drop_columns = [
    'paused', 'factor',
    'high_limit', 'low_limit',
    'sw_l1', 'sw_l3', 'sw_l2', 'zjw',
    '上海主板', '深圳主板', '科创板', '创业板', '北交所',
    'NEXT_DOJI4',
]

# TODO 分类特征。可不排序
categorical_feature = [
    # 'sw_l1',
    # 'DOJI',
]

# %%
PRED_PATH = 'pred.parquet'  # 预测结果
PRED_EXCEL = 'pred.xlsx'  # 预测结果导出Excel
