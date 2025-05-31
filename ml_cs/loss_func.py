import numpy as np
from scipy import stats


def feval_ccc(y_pred, lgb_train):
    y_true = lgb_train.get_label()

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = 2 * cov / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return 'ccc', ccc, True


def feval_pearsonr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'pearsonr', stats.pearsonr(y_true, y_pred)[0], False


def fobj_ds(y_pred, y_true):  # 损失函数
    """
    https://mp.weixin.qq.com/s/Z1yyw1BrfXcM3_G-qQ3xvw
    """
    y_true = y_true.get_label()
    residual = y_true - y_pred
    hess = np.exp(-residual)
    grad = hess - 1.0
    return grad, hess


def fscore_ds(y_pred, y_true):
    residual = y_true - y_pred
    loss = np.exp(-residual) + residual
    return np.mean(loss) - 1.0


def feval_ds(y_pred, y_true):  # 评估函数
    y_true = y_true.get_label()
    residual = y_true - y_pred
    # loss = np.e ** (-residual) + residual - 1.
    loss = np.exp(-residual) + residual
    return "ds", np.mean(loss) - 1.0, False


def feval_madl(y_pred, y_true):
    """https://mp.weixin.qq.com/s/io2JOHWLjjZi4ONdIMT7yA"""
    y_true = y_true.get_label()
    residual = y_true - y_pred
    loss = -np.sign(y_true * y_pred) * np.abs(y_true)
    return 'madl', np.mean(loss), False
