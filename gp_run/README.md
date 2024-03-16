# 遗传算法编程

本项目演示如何在遗传算法中使用表达式转代码工具。本人很早就用`deap`做过遗传算法，没用过`gplearn`，所以这次继续沿用`deap`

## 安装

```commandline
pip install -r requirements.txt # 单机运行遗传算法
pip install -r requirements_node.txt # 分布式。在head和node上都要安装
```

# 快速运行最简示例

1. 运行`data/prepare_date.py`准备数据
2. 运行`gp_run/main.py`，观察打印输出，结果生成在`log`下
3. 运行`gp_run/all_fitness.py`，得到所有样本适应度

## 本项目特点

1. 种群中有大量相似表达式，而`expr_codegen`中的`cse`公共子表达式消除可以减少大量重复计算
2. `polars`支持并发，可同一种群所有个体一起计算

所以

1. 鼓励`Rust`高手能向`Polars`贡献常用函数代码，提高效率、方便投研
2. 鼓励大家向`polars_ta`贡献代码
3. 建议使用大内存，如`>=64G`

## 目录gp_base

1. `custom.py` # 导入算子、因子、和常数
2. `deap_patch.py` # deap官方库部分要求不满足，对其动态补丁
3. `helper.py` # 一些辅助函数，部分要定制的函数也在这里

## 目录gp_run

1. `all_fitness.py` # 显示所有适应度
2. `check_exprs.py` # 当发现生成的表达式有误时，可在此对表达式进行调试。**可显示LATEX，可绘制表达式树**
3. `codegen_primitives.py` # 用于自动生成`primitives.py`
4. `primitives.py` # 自动生成的算子，仅用于参考
5. `main.py` # 单机遗传算法入口
6. `main_ray.py` # 分布式入口

## 使用进阶

1. 根据自己的需求，修改`custom.py`,添加算子、因子和常数
2. `log`目录提前备份并清空一下
3. `prepare_date.py`参考准备数据，一定要注意准备标签字段用于计算IC等指标。直接执行生成测试数据
4. `main.py`中修改遗传算法种群、代数、随机数种子等参数，运行
5. `main_ray.py`分布式版本

## 分布式运行

注意：以下都以本人Windows环境为示例。用户请根据自己实际情况进行调整

1. 每个节点python版本要一样，精确到修订版本，例如`python 3.11.7`，请在虚拟环境中使用
2. 每个节点都需要`pip install -r requirements_node.txt`。
3. 运行`main_ray.py`的节点还需要额外`pip install -r requirements.txt`
4. 将数据复制到要用来计算的节点中、不参与计算的节点不必复制。记录下对应的IP地址
5. head节点`set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1&&ray start --head --num-cpus=1`，如果head节点只是任务分发，可设置`--num-cpus=0`
6. 计算节点`set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1&&ray start --address=192.168.28.218:6379 --num-cpus=1`
7. 修改`main_ray.py`中的`ip_path`，注意：不参与计算的节点不要添加
8. 启动`main_ray.py`

### `num-cpus=1`解释

`polars`项目最大的特点是支持多线程，所以在一个节点上启动多个`actor`是没有意义的，反而可能拖慢速度。
所以在`ray start`时设置`num-cpus=1`，并且`@ray.remote(num_cpus=1)`实现一个节点上只跑一个`polars`任务

### 分布式与单机版区别

都是由一个主程序生成大量表达式，然后进行过滤去重。每一代种群的大量个体分成几批，然后分批计算。
分布式版是将几批表达式分到几个节点中进行计算，返回的几批适应度还原成一代种群

## 挖掘时序因子或横截面因子

1. 股票策略一般使用横截面相关性，计算横截面RankIC,然后计算时序ICIR等指标
2. CTA策略一般使用时序相关性，计算IC、Sharpe等

在本项目中`main.py`文件开头`from gp_base_ts`表示处理时序IC，而`from gp_base_cs`表示处理横截面IC。用户请根据自己的需求修改

## Q&A

Q: 为何生成的表达式无法直接使用?

A: 项目涉及到几个模块`sympy`、`deap`、`LaTeX`，`polars_ta`，需要取舍。以`max`为例

1. `polar_ta`，为了不与`buildins`冲突，所以命名为`max_`
2. `deap`中，为了按参数类型生成表达式更合理，所以定义了`imax(OPEN, 1)`与`fmax(OPEN, CLOSE)`
3. `deap`生成后通过`convert_inverse_prim`生成`sympy`进行简化提取公共子表达式
4. `sympy`有`Max`内部可通过`LatexPrinter`转到`LaTeX`后是`max`，`LaTeX`支持的好处是Notebook中更直观