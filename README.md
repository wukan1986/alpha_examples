# alpha_examples

本项目是将`polars_ta`, `expr_codegen`, `AlphaInspect`等几个项目利用起来的开箱即用示例

1. [polars_ta](https://github.com/wukan1986/polars_ta): 基于`polars`表达式的指标库
2. [expr_codegen](https://github.com/wukan1986/expr_codegen): 将`WorldQuant Alpha101`风格的表达式转换成`polars`风格代码的工具
3. [AlphaInspect](https://github.com/wukan1986/AlphaInspect): 仿`alphalens`的单因子分析工具

## 使用方法

```commandline
git clone --depth=1 https://github.com/wukan1986/alpha_examples.git
```

然后使用`PyCharm`或`VSCode`打开即可。

每个文件夹中的`requirements.txt`都需要安装

```commandline
pip install -r requirements.txt
```

每个文件夹下的`README.md`都请认真先看一看

注意：如果`github`无法访问，用户可以在`gitee`中新建仓库，然后导入`github`仓库即可

## data、codes、codegen 三目录

`data`生成测试用数据，它依赖于`codes`目录，而`codes`目录由`codegen`中的脚本生成

## research 单因子分析示例

可以将它做为研发单因子的模板，只要改一行因子表达式即可

1. `step1.py`: 演示准备数据
2. `step2.py`: 演示特征的研究
3. `step3.py`: 生成报表
4. `step4.py`: 修改于`step2.py`，用于生成多个特征
5. `step5.py`: 比较多个特证的相关性

## gp_base_cs/gp_base_ts/gp_run 遗传编程

自动大批量生成表达式。可将生成的表达式直接复制到`research/step2.py`、`research/step3.py`中进行进一步分析

## transformer 第三方表达转换

演示了`Alpha101`和`Alpha191`转换成`polars_ta`风格表达式