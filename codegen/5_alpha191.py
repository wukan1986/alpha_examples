import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import inspect

from expr_codegen.codes import sources_to_exprs
from expr_codegen.tool import ExprTool

# 导入OPEN等特征
from sympy_define import *  # noqa

(OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT,
 RETURNS, VWAP, CAP,
 ADV5, ADV10, ADV15, ADV20, ADV30, ADV40, ADV50, ADV60, ADV81, ADV120, ADV150, ADV180,
 SECTOR, INDUSTRY, SUBINDUSTRY,) = symbols("""OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT,
RETURNS, VWAP, CAP,
ADV5, ADV10, ADV15, ADV20, ADV30, ADV40, ADV50, ADV60, ADV81, ADV120, ADV150, ADV180,
SECTOR, INDUSTRY, SUBINDUSTRY, """, cls=Symbol)

RET, = symbols('RET, ', cls=Symbol)


# CORR = ts_corr
# RANK = cs_rank
# DELTA = ts_delta
# LOG = log
# SUM = ts_sum
# IF = if_else
# DELAY = ts_delay
# MIN = min_
# MAX = max_
# STD = ts_std_dev
# MEAN = ts_mean
# TSMAX = ts_max
# TSMIN = ts_min
# TSRANK = ts_rank
# SMA = ts_SMA_CN
# ABS = abs_
# DECAYLINEAR = ts_decay_linear
# WMA = ts_WMA
# SIGN = sign
# COUNT = ts_count
# COVIANCE = ts_covariance


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 注意：expr_codegen带算子自注册功能，但原Alpha191中所有算子都没有ts/cs前缀，会导致无法正确分组。既然要加ts/cs,还不如全部重构成wq风格

    alpha_001 = (-1 * ts_corr(cs_rank(ts_delta(log(VOLUME), 1)), cs_rank(((CLOSE - OPEN) / OPEN)), 6))
    alpha_002 = (-1 * ts_delta((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))
    alpha_003 = ts_sum(if_else(CLOSE == ts_delay(CLOSE, 1), 0, CLOSE - if_else(CLOSE > ts_delay(CLOSE, 1), min_(LOW, ts_delay(CLOSE, 1)), max_(HIGH, ts_delay(CLOSE, 1)))), 6)
    alpha_004 = if_else((((ts_sum(CLOSE, 8) / 8) + ts_std_dev(CLOSE, 8)) < (ts_sum(CLOSE, 2) / 2)), (-1 * 1), if_else(((ts_sum(CLOSE, 2) / 2) < ((ts_sum(CLOSE, 8) / 8) - ts_std_dev(CLOSE, 8))), 1, if_else(((1 < (VOLUME / ts_mean(VOLUME, 20))) | ((VOLUME / ts_mean(VOLUME, 20)) == 1)), 1, (-1 * 1))))
    alpha_005 = (-1 * ts_max(ts_corr(ts_rank(VOLUME, 5), ts_rank(HIGH, 5), 5), 3))
    alpha_006 = (cs_rank(sign(ts_delta((OPEN * 0.85 + HIGH * 0.15), 4))) * -1)
    alpha_007 = ((cs_rank(max_((VWAP - CLOSE), 3)) + cs_rank(min_((VWAP - CLOSE), 3))) * cs_rank(ts_delta(VOLUME, 3)))
    alpha_008 = cs_rank(ts_delta(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)
    alpha_009 = ts_SMA_CN(((HIGH + LOW) / 2 - (ts_delay(HIGH, 1) + ts_delay(LOW, 1)) / 2) * (HIGH - LOW) / VOLUME, 7, 2)
    alpha_010 = (cs_rank(ts_max(if_else((RET < 0), ts_std_dev(RET, 20), CLOSE) ** 2, 5)))
    alpha_011 = ts_sum(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 6)
    alpha_012 = (cs_rank((OPEN - (ts_sum(VWAP, 10) / 10)))) * (-1 * (cs_rank(abs_((CLOSE - VWAP)))))
    alpha_013 = (((HIGH * LOW) ** 0.5) - VWAP)
    alpha_014 = CLOSE - ts_delay(CLOSE, 5)
    alpha_015 = OPEN / ts_delay(CLOSE, 1) - 1
    alpha_016 = (-1 * ts_max(cs_rank(ts_corr(cs_rank(VOLUME), cs_rank(VWAP), 5)), 5))
    alpha_017 = cs_rank((VWAP - max_(VWAP, 15))) ** ts_delta(CLOSE, 5)
    alpha_018 = CLOSE / ts_delay(CLOSE, 5)
    alpha_019 = if_else(CLOSE < ts_delay(CLOSE, 5), (CLOSE - ts_delay(CLOSE, 5)) / ts_delay(CLOSE, 5), if_else(CLOSE == ts_delay(CLOSE, 5), 0, (CLOSE - ts_delay(CLOSE, 5)) / CLOSE))
    alpha_020 = (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE, 6) * 100
    # alpha_021 = REGSLOPE(MEAN(CLOSE, 6), 6)
    alpha_022 = ts_SMA_CN(((CLOSE - ts_mean(CLOSE, 6)) / ts_mean(CLOSE, 6) - ts_delay((CLOSE - ts_mean(CLOSE, 6)) / ts_mean(CLOSE, 6), 3)), 12, 1)
    alpha_023 = ts_SMA_CN(if_else(CLOSE > ts_delay(CLOSE, 1), ts_std_dev(CLOSE, 20), 0), 20, 1) / (
            ts_SMA_CN(if_else(CLOSE > ts_delay(CLOSE, 1), ts_std_dev(CLOSE, 20), 0),
                      20, 1) + ts_SMA_CN(if_else(CLOSE <= ts_delay(CLOSE, 1), ts_std_dev(CLOSE, 20), 0), 20, 1)) * 100
    alpha_024 = ts_SMA_CN(CLOSE - ts_delay(CLOSE, 5), 5, 1)
    alpha_025 = ((-1 * cs_rank((ts_delta(CLOSE, 7) * (1 - cs_rank(ts_decay_linear((VOLUME / ts_mean(VOLUME, 20)), 9)))))) * (1 +
                                                                                                                             cs_rank(ts_sum(RET,
                                                                                                                                            250))))
    alpha_026 = (((ts_sum(CLOSE, 7) / 7) - CLOSE) + (ts_corr(VWAP, ts_delay(CLOSE, 5), 230)))
    alpha_027 = ts_WMA((CLOSE - ts_delay(CLOSE, 3)) / ts_delay(CLOSE, 3) * 100 + (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE, 6) * 100,
                       12)
    alpha_028 = 3 * ts_SMA_CN((CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100, 3, 1) - 2 * ts_SMA_CN(
        ts_SMA_CN((CLOSE - ts_min(LOW, 9)) / (
                max_(HIGH, 9) - ts_max(LOW, 9)) * 100, 3, 1), 3, 1)
    alpha_029 = (CLOSE - ts_delay(CLOSE, 6)) / ts_delay(CLOSE, 6) * VOLUME
    # alpha_030 = WMA((REGRESI(CLOSE / DELAY(CLOSE, 1) - 1, MKT, SMB, HML, 60)) ** 2, 20)
    alpha_031 = (CLOSE - ts_mean(CLOSE, 12)) / ts_mean(CLOSE, 12) * 100
    alpha_032 = (-1 * ts_sum(cs_rank(ts_corr(cs_rank(HIGH), cs_rank(VOLUME), 3)), 3))
    alpha_033 = ((((-1 * ts_min(LOW, 5)) + ts_delay(ts_min(LOW, 5), 5)) * cs_rank(((ts_sum(RET, 240) - ts_sum(RET, 20)) / 220))) *
                 ts_rank(VOLUME, 5))
    alpha_034 = ts_mean(CLOSE, 12) / CLOSE
    alpha_035 = (min_(cs_rank(ts_decay_linear(ts_delta(OPEN, 1), 15)), cs_rank(ts_decay_linear(ts_corr(VOLUME, ((OPEN * 0.65) +
                                                                                                                (OPEN * 0.35)), 17), 7))) * -1)
    alpha_036 = cs_rank(ts_sum(ts_corr(cs_rank(VOLUME), cs_rank(VWAP), 6), 2))
    alpha_037 = (-1 * cs_rank(((ts_sum(OPEN, 5) * ts_sum(RET, 5)) - ts_delay((ts_sum(OPEN, 5) * ts_sum(RET, 5)), 10))))
    alpha_038 = if_else(((ts_sum(HIGH, 20) / 20) < HIGH), (-1 * ts_delta(HIGH, 2)), 0)
    alpha_039 = ((cs_rank(ts_decay_linear(ts_delta(CLOSE, 2), 8)) - cs_rank(ts_decay_linear(ts_corr(((VWAP * 0.3) + (OPEN * 0.7)),
                                                                                                    ts_sum(ts_mean(VOLUME, 180), 37), 14),
                                                                                            12))) * -1)
    alpha_040 = ts_sum(if_else(CLOSE > ts_delay(CLOSE, 1), VOLUME, 0), 26) / ts_sum(if_else(CLOSE <= ts_delay(CLOSE, 1), VOLUME, 0), 26) * 100
    alpha_041 = (cs_rank(max_(ts_delta(VWAP, 3), 5)) * -1)
    alpha_042 = ((-1 * cs_rank(ts_std_dev(HIGH, 10))) * ts_corr(HIGH, VOLUME, 10))
    alpha_043 = ts_sum(if_else(CLOSE > ts_delay(CLOSE, 1), VOLUME, if_else(CLOSE < ts_delay(CLOSE, 1), -VOLUME, 0)), 6)
    alpha_044 = (ts_rank(ts_decay_linear(ts_corr((LOW), ts_mean(VOLUME, 10), 7), 6), 4) + ts_rank(ts_decay_linear(ts_delta((VWAP),
                                                                                                                           3), 10), 15))
    alpha_045 = (cs_rank(ts_delta((CLOSE * 0.6 + OPEN * 0.4), 1)) * cs_rank(ts_corr(VWAP, ts_mean(VOLUME, 150), 15)))
    alpha_046 = (ts_mean(CLOSE, 3) + ts_mean(CLOSE, 6) + ts_mean(CLOSE, 12) + ts_mean(CLOSE, 24)) / (4 * CLOSE)
    alpha_047 = ts_SMA_CN((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 9, 1)
    alpha_048 = (-1 * ((cs_rank(((sign((CLOSE - ts_delay(CLOSE, 1))) + sign((ts_delay(CLOSE, 1) - ts_delay(CLOSE, 2)))) +
                                 sign((ts_delay(CLOSE, 2) - ts_delay(CLOSE, 3)))))) * ts_sum(VOLUME, 5)) / ts_sum(VOLUME, 20))
    alpha_049 = ts_sum(if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                               max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) / (ts_sum(
        if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) + ts_sum(
        if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12))
    alpha_050 = ts_sum(if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                               max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) / (ts_sum(
        if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) + ts_sum(
        if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12)) - ts_sum(
        if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) / (ts_sum(
        if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) + ts_sum(
        if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12))
    alpha_051 = ts_sum(if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                               max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) / (ts_sum(
        if_else((HIGH + LOW) <= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12) + ts_sum(
        if_else((HIGH + LOW) >= (ts_delay(HIGH, 1) + ts_delay(LOW, 1)), 0,
                max_(abs_(HIGH - ts_delay(HIGH, 1)), abs_(LOW - ts_delay(LOW, 1)))), 12))
    alpha_052 = ts_sum(max_(0, HIGH - ts_delay((HIGH + LOW + CLOSE) / 3, 1)), 26) / ts_sum(
        max_(0, ts_delay((HIGH + LOW + CLOSE) / 3, 1) - LOW), 26) * 100
    alpha_053 = ts_count(CLOSE > ts_delay(CLOSE, 1), 12) / 12 * 100
    alpha_054 = (-1 * cs_rank((ts_std_dev(abs_(CLOSE - OPEN), 5) + (CLOSE - OPEN)) + ts_corr(CLOSE, OPEN, 10)))
    alpha_055 = ts_sum(16 * (CLOSE - ts_delay(CLOSE, 1) + (CLOSE - OPEN) / 2 + ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / (if_else(
        (abs_(HIGH - ts_delay(CLOSE, 1)) > abs_(LOW - ts_delay(CLOSE, 1))) & (
                abs_(HIGH - ts_delay(CLOSE, 1)) > abs_(HIGH - ts_delay(LOW, 1))),
        abs_(HIGH - ts_delay(CLOSE, 1)) + abs_(LOW - ts_delay(CLOSE, 1)) / 2 + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4, if_else(
            (abs_(LOW - ts_delay(CLOSE, 1)) > abs_(HIGH - ts_delay(LOW, 1))) & (
                    abs_(LOW - ts_delay(CLOSE, 1)) > abs_(HIGH - ts_delay(CLOSE, 1))),
            abs_(LOW - ts_delay(CLOSE, 1)) + abs_(HIGH - ts_delay(CLOSE, 1)) / 2 + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4,
            abs_(HIGH - ts_delay(LOW, 1)) + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4))) * max_(abs_(HIGH - ts_delay(CLOSE, 1)),
                                                                                                       abs_(LOW - ts_delay(CLOSE, 1))),
                       20)
    alpha_056 = (cs_rank((OPEN - ts_min(OPEN, 12))) < cs_rank((cs_rank(ts_corr(ts_sum(((HIGH + LOW) / 2), 19), ts_sum(ts_mean(VOLUME, 40), 19), 13)) ** 5)))
    alpha_057 = ts_SMA_CN((CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100, 3, 1)
    alpha_058 = ts_count(CLOSE > ts_delay(CLOSE, 1), 20) / 20 * 100
    alpha_059 = ts_sum(if_else(CLOSE == ts_delay(CLOSE, 1), 0, CLOSE - if_else(CLOSE > ts_delay(CLOSE, 1), min_(LOW, ts_delay(CLOSE, 1)), max_(HIGH, ts_delay(CLOSE, 1)))), 20)
    alpha_060 = ts_sum(((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW) * VOLUME, 20)
    alpha_061 = (max_(cs_rank(ts_decay_linear(ts_delta(VWAP, 1), 12)),
                      cs_rank(ts_decay_linear(cs_rank(ts_corr(LOW, ts_mean(VOLUME, 80), 8)), 17))) * -1)
    alpha_062 = (-1 * ts_corr(HIGH, cs_rank(VOLUME), 5))
    alpha_063 = ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 6, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 6, 1) * 100
    alpha_064 = (max_(cs_rank(ts_decay_linear(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 4), 4)),
                      cs_rank(ts_decay_linear(max_(ts_corr(cs_rank(CLOSE), cs_rank(ts_mean(VOLUME, 60)), 4), 13), 14))) * -1)
    alpha_065 = ts_mean(CLOSE, 6) / CLOSE
    alpha_066 = (CLOSE - ts_mean(CLOSE, 6)) / ts_mean(CLOSE, 6) * 100
    alpha_067 = ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 24, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 24, 1) * 100
    alpha_068 = ts_SMA_CN(((HIGH + LOW) / 2 - (ts_delay(HIGH, 1) + ts_delay(LOW, 1)) / 2) * (HIGH - LOW) / VOLUME, 15, 2)
    # alpha_069 = IF(SUM(DTM, 20) > SUM(DBM, 20), (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DTM, 20), IF(SUM(DTM, 20) == SUM(DBM, 20), 0, (SUM(DTM, 20) - SUM(DBM, 20)) / SUM(DBM, 20)))
    alpha_070 = ts_std_dev(AMOUNT, 6)
    alpha_071 = (CLOSE - ts_mean(CLOSE, 24)) / ts_mean(CLOSE, 24) * 100
    alpha_072 = ts_SMA_CN((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 15, 1)
    alpha_073 = ((ts_rank(ts_decay_linear(ts_decay_linear(ts_corr(CLOSE, VOLUME, 10), 16), 4), 5) -
                  cs_rank(ts_decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 30), 4), 3))) * -1)
    alpha_074 = (cs_rank(ts_corr(ts_sum(((LOW * 0.35) + (VWAP * 0.65)), 20), ts_sum(ts_mean(VOLUME, 40), 20), 7)) +
                 cs_rank(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 6)))
    # alpha_075 = COUNT((CLOSE > OPEN) & (BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN), 50) / COUNT(BANCHMARKINDEXCLOSE < BANCHMARKINDEXOPEN, 50)
    alpha_076 = ts_std_dev(abs_((CLOSE / ts_delay(CLOSE, 1) - 1)) / VOLUME, 20) / ts_mean(abs_((CLOSE / ts_delay(CLOSE, 1) - 1)) / VOLUME, 20)
    alpha_077 = min_(cs_rank(ts_decay_linear(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),
                     cs_rank(ts_decay_linear(ts_corr(((HIGH + LOW) / 2), ts_mean(VOLUME, 40), 3), 6)))
    alpha_078 = ((HIGH + LOW + CLOSE) / 3 - ts_mean((HIGH + LOW + CLOSE) / 3, 12)) / (0.015 * ts_mean(ABS(CLOSE - ts_mean((HIGH + LOW + CLOSE) / 3, 12)), 12))
    alpha_079 = ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 12, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 12, 1) * 100
    alpha_080 = (VOLUME - ts_delay(VOLUME, 5)) / ts_delay(VOLUME, 5) * 100
    alpha_081 = ts_SMA_CN(VOLUME, 21, 2)
    alpha_082 = ts_SMA_CN((ts_max(HIGH, 6) - CLOSE) / (ts_max(HIGH, 6) - ts_min(LOW, 6)) * 100, 20, 1)
    alpha_083 = (-1 * cs_rank(ts_covariance(cs_rank(HIGH), cs_rank(VOLUME), 5)))
    alpha_084 = ts_sum(if_else(CLOSE > ts_delay(CLOSE, 1), VOLUME, if_else(CLOSE < ts_delay(CLOSE, 1), -VOLUME, 0)), 20)
    alpha_085 = (ts_rank((VOLUME / ts_mean(VOLUME, 20)), 20) * ts_rank((-1 * ts_delta(CLOSE, 7)), 8))
    alpha_086 = if_else((0.25 < (((ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10) - ((ts_delay(CLOSE, 10) - CLOSE) / 10))), (-1 * 1),
                        if_else(((((ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10) - ((ts_delay(CLOSE, 10) - CLOSE) / 10)) < 0), 1,
                                ((-1 * 1) *
                                 (CLOSE - ts_delay(CLOSE, 1)))))
    alpha_087 = ((cs_rank(ts_decay_linear(ts_delta(VWAP, 4), 7)) + ts_rank(ts_decay_linear(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /
                                                                                            (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
    alpha_088 = (CLOSE - ts_delay(CLOSE, 20)) / ts_delay(CLOSE, 20) * 100
    alpha_089 = 2 * (ts_SMA_CN(CLOSE, 13, 2) - ts_SMA_CN(CLOSE, 27, 2) - ts_SMA_CN(ts_SMA_CN(CLOSE, 13, 2) - ts_SMA_CN(CLOSE, 27, 2), 10, 2))
    alpha_090 = (cs_rank(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 5)) * -1)
    alpha_091 = ((cs_rank((CLOSE - max_(CLOSE, 5))) * cs_rank(ts_corr((ts_mean(VOLUME, 40)), LOW, 5))) * -1)
    alpha_092 = (max_(cs_rank(ts_decay_linear(ts_delta(((CLOSE * 0.35) + (VWAP * 0.65)), 2), 3)),
                      ts_rank(ts_decay_linear(abs_(ts_corr((ts_mean(VOLUME, 180)), CLOSE, 13)), 5), 15)) * -1)
    alpha_093 = ts_sum(if_else(OPEN >= ts_delay(OPEN, 1), 0, max_((OPEN - LOW), (OPEN - ts_delay(OPEN, 1)))), 20)
    alpha_094 = ts_sum(if_else(CLOSE > ts_delay(CLOSE, 1), VOLUME, if_else(CLOSE < ts_delay(CLOSE, 1), -VOLUME, 0)), 30)
    alpha_095 = ts_std_dev(AMOUNT, 20)
    alpha_096 = ts_SMA_CN(ts_SMA_CN((CLOSE - ts_min(LOW, 9)) / (ts_max(HIGH, 9) - ts_min(LOW, 9)) * 100, 3, 1), 3, 1)
    alpha_097 = ts_std_dev(VOLUME, 10)
    alpha_098 = if_else(
        (((ts_delta((ts_sum(CLOSE, 100) / 100), 100) / ts_delay(CLOSE, 100)) < 0.05) | ((ts_delta((ts_sum(CLOSE, 100) / 100), 100) /
                                                                                         ts_delay(CLOSE, 100)) == 0.05)),
        (-1 * (CLOSE - ts_min(CLOSE, 100))), (-1 * ts_delta(CLOSE, 3)))
    alpha_099 = (-1 * cs_rank(ts_covariance(cs_rank(CLOSE), cs_rank(VOLUME), 5)))
    alpha_100 = ts_std_dev(VOLUME, 20)
    alpha_101 = ((cs_rank(ts_corr(CLOSE, ts_sum(ts_mean(VOLUME, 30), 37), 15)) < cs_rank(ts_corr(cs_rank(((HIGH * 0.1) + (VWAP * 0.9))), cs_rank(VOLUME), 11))) * -1)
    alpha_102 = ts_SMA_CN(max_(VOLUME - ts_delay(VOLUME, 1), 0), 6, 1) / ts_SMA_CN(abs_(VOLUME - ts_delay(VOLUME, 1)), 6, 1) * 100
    alpha_103 = ((20 - ts_arg_min(LOW, 20)) / 20) * 100
    alpha_104 = (-1 * (ts_delta(ts_corr(HIGH, VOLUME, 5), 5) * cs_rank(ts_std_dev(CLOSE, 20))))
    alpha_105 = (-1 * ts_corr(cs_rank(OPEN), cs_rank(VOLUME), 10))
    alpha_106 = CLOSE - ts_delay(CLOSE, 20)
    alpha_107 = (((-1 * cs_rank((OPEN - ts_delay(HIGH, 1)))) * cs_rank((OPEN - ts_delay(CLOSE, 1)))) * cs_rank((OPEN - ts_delay(LOW, 1))))
    alpha_108 = ((cs_rank((HIGH - min_(HIGH, 2))) ** cs_rank(ts_corr(VWAP, (ts_mean(VOLUME, 120)), 6))) * -1)
    alpha_109 = ts_SMA_CN(HIGH - LOW, 10, 2) / ts_SMA_CN(ts_SMA_CN(HIGH - LOW, 10, 2), 10, 2)
    alpha_110 = ts_sum(max_(0, HIGH - ts_delay(CLOSE, 1)), 20) / ts_sum(max_(0, ts_delay(CLOSE, 1) - LOW), 20) * 100
    alpha_111 = ts_SMA_CN(VOLUME * ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW), 11, 2) - ts_SMA_CN(
        VOLUME * ((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW), 4, 2)
    alpha_112 = (ts_sum(if_else(CLOSE - ts_delay(CLOSE, 1) > 0, CLOSE - ts_delay(CLOSE, 1), 0), 12) - ts_sum(
        if_else(CLOSE - ts_delay(CLOSE, 1) < 0, abs_(CLOSE - ts_delay(CLOSE, 1)), 0), 12)) / (
                        ts_sum(if_else(CLOSE - ts_delay(CLOSE, 1) > 0, CLOSE - ts_delay(CLOSE, 1), 0), 12) + ts_sum(
                    if_else(CLOSE - ts_delay(CLOSE, 1) < 0, abs_(CLOSE - ts_delay(CLOSE, 1)), 0), 12)) * 100
    alpha_113 = (-1 * ((cs_rank((ts_sum(ts_delay(CLOSE, 5), 20) / 20)) * ts_corr(CLOSE, VOLUME, 2)) * cs_rank(
        ts_corr(ts_sum(CLOSE, 5), ts_sum(CLOSE, 20), 2))))
    alpha_114 = ((cs_rank(ts_delay(((HIGH - LOW) / (ts_sum(CLOSE, 5) / 5)), 2)) * cs_rank(cs_rank(VOLUME))) / (((HIGH - LOW) /
                                                                                                                (ts_sum(CLOSE, 5) / 5)) / (
                                                                                                                       VWAP - CLOSE)))
    alpha_115 = (cs_rank(ts_corr(((HIGH * 0.9) + (CLOSE * 0.1)), ts_mean(VOLUME, 30), 10)) ** cs_rank(ts_corr(ts_rank(((HIGH + LOW) /
                                                                                                                       2), 4),
                                                                                                              ts_rank(VOLUME, 10), 7)))
    # alpha_116 = REGSLOPE(CLOSE, 20)
    alpha_117 = ((ts_rank(VOLUME, 32) * (1 - ts_rank(((CLOSE + HIGH) - LOW), 16))) * (1 - ts_rank(RET, 32)))
    alpha_118 = ts_sum(HIGH - OPEN, 20) / ts_sum(OPEN - LOW, 20) * 100
    alpha_119 = (cs_rank(ts_decay_linear(ts_corr(VWAP, ts_sum(ts_mean(VOLUME, 5), 26), 5), 7)) -
                 cs_rank(ts_decay_linear(ts_rank(min_(ts_corr(cs_rank(OPEN), cs_rank(ts_mean(VOLUME, 15)), 21), 9), 7), 8)))
    alpha_120 = (cs_rank((VWAP - CLOSE)) / cs_rank((VWAP + CLOSE)))
    alpha_121 = ((cs_rank((VWAP - min_(VWAP, 12))) ** ts_rank(ts_corr(ts_rank(VWAP, 20), ts_rank(ts_mean(VOLUME, 60), 2), 18), 3)) * -1)
    alpha_122 = (ts_SMA_CN(ts_SMA_CN(ts_SMA_CN(log(CLOSE), 13, 2), 13, 2), 13, 2) - ts_delay(ts_SMA_CN(ts_SMA_CN(ts_SMA_CN(log(CLOSE), 13, 2), 13, 2), 13, 2),
                                                                                             1)) / ts_delay(
        ts_SMA_CN(ts_SMA_CN(ts_SMA_CN(log(CLOSE), 13, 2), 13, 2), 13, 2), 1)
    alpha_123 = ((cs_rank(ts_corr(ts_sum(((HIGH + LOW) / 2), 20), ts_sum(ts_mean(VOLUME, 60), 20), 9)) < cs_rank(ts_corr(LOW, VOLUME, 6))) * -1)
    alpha_124 = (CLOSE - VWAP) / ts_decay_linear(cs_rank(ts_max(CLOSE, 30)), 2)
    alpha_125 = (cs_rank(ts_decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 80), 17), 20)) / cs_rank(ts_decay_linear(ts_delta(((CLOSE * 0.5)
                                                                                                                          + (VWAP * 0.5)), 3), 16)))
    alpha_126 = (CLOSE + HIGH + LOW) / 3
    alpha_127 = (ts_mean((100 * (CLOSE - ts_max(CLOSE, 12)) / (ts_max(CLOSE, 12))) ** 2, 12)) ** (1 / 2)
    alpha_128 = 100 - (100 / (1 + ts_sum(
        if_else((HIGH + LOW + CLOSE) / 3 > ts_delay((HIGH + LOW + CLOSE) / 3, 1), (HIGH + LOW + CLOSE) / 3 * VOLUME, 0),
        14) / ts_sum(
        if_else((HIGH + LOW + CLOSE) / 3 < ts_delay((HIGH + LOW + CLOSE) / 3, 1), (HIGH + LOW + CLOSE) / 3 * VOLUME, 0), 14)))
    alpha_129 = ts_sum(if_else(CLOSE - ts_delay(CLOSE, 1) < 0, abs_(CLOSE - ts_delay(CLOSE, 1)), 0), 12)
    alpha_130 = (cs_rank(ts_decay_linear(ts_corr(((HIGH + LOW) / 2), ts_mean(VOLUME, 40), 9), 10)) /
                 cs_rank(ts_decay_linear(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 7), 3)))
    alpha_131 = (cs_rank(ts_delta(VWAP, 1)) ** ts_rank(ts_corr(CLOSE, ts_mean(VOLUME, 50), 18), 18))
    alpha_132 = ts_mean(AMOUNT, 20)
    alpha_133 = ((20 - ts_arg_max(HIGH, 20)) / 20) * 100 - ((20 - ts_arg_min(LOW, 20)) / 20) * 100
    alpha_134 = (CLOSE - ts_delay(CLOSE, 12)) / ts_delay(CLOSE, 12) * VOLUME
    alpha_135 = ts_SMA_CN(ts_delay(CLOSE / ts_delay(CLOSE, 20), 1), 20, 1)
    alpha_136 = ((-1 * cs_rank(ts_delta(RET, 3))) * ts_corr(OPEN, VOLUME, 10))
    alpha_137 = 16 * (CLOSE - ts_delay(CLOSE, 1) + (CLOSE - OPEN) / 2 + ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / (if_else(
        (abs_(HIGH - ts_delay(CLOSE, 1)) > abs_(LOW - ts_delay(CLOSE, 1))) & (abs_(HIGH - ts_delay(CLOSE, 1)) > abs_(
            HIGH - ts_delay(LOW, 1))),
        abs_(HIGH - ts_delay(CLOSE, 1)) + abs_(LOW - ts_delay(CLOSE, 1)) / 2 + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4, if_else(
            (abs_(LOW - ts_delay(CLOSE, 1)) > abs_(HIGH - ts_delay(LOW, 1))) & (abs_(LOW - ts_delay(CLOSE, 1)) > abs_(
                HIGH - ts_delay(CLOSE, 1))),
            abs_(LOW - ts_delay(CLOSE, 1)) + abs_(HIGH - ts_delay(CLOSE, 1)) / 2 + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4,
            abs_(HIGH - ts_delay(LOW, 1)) + abs_(ts_delay(CLOSE, 1) - ts_delay(OPEN, 1)) / 4))) * max_(abs_(HIGH - ts_delay(CLOSE, 1)),
                                                                                                       abs_(LOW - ts_delay(CLOSE, 1)))
    alpha_138 = ((cs_rank(ts_decay_linear(ts_delta((LOW * 0.7 + VWAP * 0.3), 3), 20)) -
                  ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(LOW, 8), ts_rank(ts_mean(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
    alpha_139 = (-1 * ts_corr(OPEN, VOLUME, 10))

    alpha_140 = min_(cs_rank(ts_decay_linear(((cs_rank(OPEN) + cs_rank(LOW)) - (cs_rank(HIGH) + cs_rank(CLOSE))), 8)),
                     ts_rank(ts_decay_linear(ts_corr(ts_rank(CLOSE, 8), ts_rank(ts_mean(VOLUME, 60), 20), 8), 7), 3))
    alpha_141 = (cs_rank(ts_corr(cs_rank(HIGH), cs_rank(ts_mean(VOLUME, 15)), 9)) * -1)
    alpha_142 = (((-1 * cs_rank(ts_rank(CLOSE, 10))) * cs_rank(ts_delta(ts_delta(CLOSE, 1), 1))) * cs_rank(ts_rank((VOLUME
                                                                                                                    / ts_mean(VOLUME, 20)), 5)))
    # alpha_143 = CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF
    # alpha_144 = SUMIF(abs_(CLOSE / ts_delay(CLOSE, 1) - 1) / AMOUNT, CLOSE < ts_delay(CLOSE, 1), 20) / ts_count(CLOSE < ts_delay(CLOSE, 1), 20)
    alpha_145 = (ts_mean(VOLUME, 9) - ts_mean(VOLUME, 26)) / ts_mean(VOLUME, 12) * 100
    alpha_146 = ts_mean((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1) - ts_SMA_CN((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1), 61, 2),
                        20) * ((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1) - ts_SMA_CN((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1),
                                                                                             61, 2)) / ts_SMA_CN(((CLOSE - ts_delay(CLOSE,
                                                                                                                                    1)) / ts_delay(
        CLOSE, 1) - ((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1) - ts_SMA_CN((CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1), 61,
                                                                                   2))) ** 2, 60, 1)
    # alpha_147 = REGSLOPE(MEAN(CLOSE, 12), 12)
    alpha_148 = ((cs_rank(ts_corr((OPEN), ts_sum(ts_mean(VOLUME, 60), 9), 6)) < cs_rank((OPEN - ts_min(OPEN, 14)))) * -1)
    # alpha_149 = REGBETA(FILTER(CLOSE / DELAY(CLOSE, 1) - 1, BANCHMARKINDEXCLOSE < DELAY(BANCHMARKINDEXCLOSE, 1)), FILTER(BANCHMARKINDEXCLOSE / DELAY(BANCHMARKINDEXCLOSE, 1) - 1,BANCHMARKINDEXCLOSE < DELAY(BANCHMARKINDEXCLOSE, 1)), 252)

    alpha_150 = (CLOSE + HIGH + LOW) / 3 * VOLUME
    alpha_151 = ts_SMA_CN(CLOSE - ts_delay(CLOSE, 20), 20, 1)
    alpha_152 = ts_SMA_CN(ts_mean(ts_delay(ts_SMA_CN(ts_delay(CLOSE / ts_delay(CLOSE, 9), 1), 9, 1), 1), 12) - ts_mean(
        ts_delay(ts_SMA_CN(ts_delay(CLOSE / ts_delay(CLOSE, 9), 1), 9, 1), 1), 26), 9, 1)
    alpha_153 = (ts_mean(CLOSE, 3) + ts_mean(CLOSE, 6) + ts_mean(CLOSE, 12) + ts_mean(CLOSE, 24)) / 4
    alpha_154 = ((VWAP - min_(VWAP, 16)) < (ts_corr(VWAP, ts_mean(VOLUME, 180), 18)))
    alpha_155 = ts_SMA_CN(VOLUME, 13, 2) - ts_SMA_CN(VOLUME, 27, 2) - ts_SMA_CN(ts_SMA_CN(VOLUME, 13, 2) - ts_SMA_CN(VOLUME, 27, 2), 10, 2)
    alpha_156 = (max_(cs_rank(ts_decay_linear(ts_delta(VWAP, 5), 3)), cs_rank(ts_decay_linear(((ts_delta(((OPEN * 0.15) + (LOW * 0.85)),
                                                                                                         2) / ((OPEN * 0.15) + (
            LOW * 0.85))) * -1), 3))) * -1)
    # alpha_157 = (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))
    alpha_158 = ((HIGH - ts_SMA_CN(CLOSE, 15, 2)) - (LOW - ts_SMA_CN(CLOSE, 15, 2))) / CLOSE
    alpha_159 = ((CLOSE - ts_sum(min_(LOW, ts_delay(CLOSE, 1)), 6)) / ts_sum(max_(HIGH, ts_delay(CLOSE, 1)) - min_(LOW, ts_delay(CLOSE, 1)), 6)
                 * 12 * 24 + (CLOSE - ts_sum(min_(LOW, ts_delay(CLOSE, 1)), 12)) / ts_sum(
                max_(HIGH, ts_delay(CLOSE, 1)) - min_(LOW, ts_delay(CLOSE, 1)), 12) * 6 * 24 + (
                         CLOSE - ts_sum(min_(LOW, ts_delay(CLOSE, 1)), 24)) / ts_sum(
                max_(HIGH, ts_delay(CLOSE, 1)) - min_(LOW, ts_delay(CLOSE, 1)), 24) * 6 * 24) * 100 / (
                        6 * 12 + 6 * 24 + 12 * 24)

    alpha_160 = ts_SMA_CN(if_else(CLOSE <= ts_delay(CLOSE, 1), ts_std_dev(CLOSE, 20), 0), 20, 1)
    alpha_161 = ts_mean(max_(max_((HIGH - LOW), abs_(ts_delay(CLOSE, 1) - HIGH)), abs_(ts_delay(CLOSE, 1) - LOW)), 12)
    alpha_162 = (ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 12, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 12, 1) * 100 - ts_min(
        ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 12, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 12, 1) * 100, 12)) / (
                        ts_max(ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 12, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 12, 1) * 100,
                               12) - ts_min(
                    ts_SMA_CN(max_(CLOSE - ts_delay(CLOSE, 1), 0), 12, 1) / ts_SMA_CN(abs_(CLOSE - ts_delay(CLOSE, 1)), 12, 1) * 100, 12))
    alpha_163 = cs_rank(((((-1 * RET) * ts_mean(VOLUME, 20)) * VWAP) * (HIGH - CLOSE)))
    alpha_164 = ts_SMA_CN((if_else((CLOSE > ts_delay(CLOSE, 1)), 1 / (CLOSE - ts_delay(CLOSE, 1)), 1) - min_(
        if_else((CLOSE > ts_delay(CLOSE, 1)), 1 / (CLOSE - ts_delay(CLOSE, 1)), 1), 12)) / (HIGH - LOW) * 100, 13, 2)
    # alpha_165 = MAX(SUMAC(CLOSE - MEAN(CLOSE, 48))) - MIN(SUMAC(CLOSE - MEAN(CLOSE, 48))) / STD(CLOSE, 48)
    alpha_166 = -20 * (20 - 1) ** 1.5 * ts_sum(CLOSE / ts_delay(CLOSE, 1) - 1 - ts_mean(CLOSE / ts_delay(CLOSE, 1) - 1, 20), 20) / (
            (20 - 1) * (20 - 2) * (ts_sum((CLOSE / ts_delay(CLOSE, 1)) ** 2, 20)) ** 1.5)
    alpha_167 = ts_sum(if_else(CLOSE - ts_delay(CLOSE, 1) > 0, CLOSE - ts_delay(CLOSE, 1), 0), 12)
    alpha_168 = (-1 * VOLUME / ts_mean(VOLUME, 20))
    alpha_169 = ts_SMA_CN(
        ts_mean(ts_delay(ts_SMA_CN(CLOSE - ts_delay(CLOSE, 1), 9, 1), 1), 12) - ts_mean(ts_delay(ts_SMA_CN(CLOSE - ts_delay(CLOSE, 1), 9, 1), 1), 26),
        10, 1)

    alpha_170 = ((((cs_rank((1 / CLOSE)) * VOLUME) / ts_mean(VOLUME, 20)) * ((HIGH * cs_rank((HIGH - CLOSE))) / (ts_sum(HIGH, 5) / 5))) - cs_rank((VWAP - ts_delay(VWAP, 5))))
    alpha_171 = ((-1 * ((LOW - CLOSE) * (OPEN ** 5))) / ((CLOSE - HIGH) * (CLOSE ** 5)))
    # alpha_172 =
    alpha_173 = 3 * ts_SMA_CN(CLOSE, 13, 2) - 2 * ts_SMA_CN(ts_SMA_CN(CLOSE, 13, 2), 13, 2) + ts_SMA_CN(ts_SMA_CN(ts_SMA_CN(log(CLOSE), 13, 2), 13, 2), 13, 2)
    alpha_174 = ts_SMA_CN(if_else(CLOSE > ts_delay(CLOSE, 1), ts_std_dev(CLOSE, 20), 0), 20, 1)
    alpha_175 = ts_mean(max_(max_((HIGH - LOW), abs_(ts_delay(CLOSE, 1) - HIGH)), abs_(ts_delay(CLOSE, 1) - LOW)), 6)
    alpha_176 = ts_corr(cs_rank(((CLOSE - ts_min(LOW, 12)) / (ts_max(HIGH, 12) - ts_min(LOW, 12)))), cs_rank(VOLUME), 6)
    alpha_177 = ((20 - ts_arg_max(HIGH, 20)) / 20) * 100
    alpha_178 = (CLOSE - ts_delay(CLOSE, 1)) / ts_delay(CLOSE, 1) * VOLUME
    alpha_179 = (cs_rank(ts_corr(VWAP, VOLUME, 4)) * cs_rank(ts_corr(cs_rank(LOW), cs_rank(ts_mean(VOLUME, 50)), 12)))

    alpha_180 = if_else((ts_mean(VOLUME, 20) < VOLUME),
                        (-1 * ts_rank(abs_(ts_delta(CLOSE, 7)), 60)) * sign(ts_delta(CLOSE, 7)),
                        (-1 * VOLUME))
    # alpha_181 = SUM(((CLOSE / DELAY(CLOSE, 1) - 1) - MEAN((CLOSE / DELAY(CLOSE, 1) - 1), 20)) - (BANCHMARKINDEXCLOSE - MEAN(BANCHMARKINDEXCLOSE, 20)) ** 2, 20) / SUM((BANCHMARKINDEXCLOSE - MEAN(BANCHMARKINDEXCLOSE, 20)) ** 3, 20)
    # alpha_182 =
    # alpha_183 = MAX(SUMAC(CLOSE - MEAN(CLOSE, 24))) - MIN(SUMAC(CLOSE - MEAN(CLOSE, 24))) / STD(CLOSE, 24)
    alpha_184 = (cs_rank(ts_corr(ts_delay((OPEN - CLOSE), 1), CLOSE, 200)) + cs_rank((OPEN - CLOSE)))
    alpha_185 = cs_rank((-1 * ((1 - (OPEN / CLOSE)) ** 2)))
    # alpha_186 =
    alpha_187 = ts_sum(if_else(OPEN <= ts_delay(OPEN, 1), 0, max_((HIGH - OPEN), (OPEN - ts_delay(OPEN, 1)))), 20)
    alpha_188 = ((HIGH - LOW - ts_SMA_CN(HIGH - LOW, 11, 2)) / ts_SMA_CN(HIGH - LOW, 11, 2)) * 100
    alpha_189 = ts_mean(abs_(CLOSE - ts_mean(CLOSE, 6)), 6)

    # alpha_190 =
    alpha_191 = ((ts_corr(ts_mean(VOLUME, 20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)


# 读取源代码，转成字符串
source = inspect.getsource(_code_block_)
raw, exprs_dict = sources_to_exprs(globals().copy(), source)
# 生成代码
tool = ExprTool()
codes, G = tool.all(exprs_dict, style='polars', template_file='template.py.j2',
                    replace=True, regroup=True, format=True,
                    date='date', asset='asset',
                    # 还复制了最原始的表达式
                    extra_codes=())

print(codes)
#
# 保存代码到指定文件
output_file = 'codes/alpha191.py'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(codes)
