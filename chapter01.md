## 怎么样的数据
主要指的是结构化数据（structured data）例如：

表格型数据，其中各列可能是不同的类型（字符串、数值、日期等）。比如保存在关系型数据库中或以制表符/逗号为分隔符的文本文件中的那些数据。
多维数组（矩阵）。
通过关键列（对于 SQL 用户而言，就是主键和外键）相互联系的多个表。
间隔平均或不平均的时间序列。
#### 大部分数据集都能被转化为更加适合分析和建模的结构化形式，虽然有时这并不是很明显。如果不行的话，也可以将数据集的特征提取为某种结构化形式。

### 重要的python库
#### numpy
快速高效的多维数组对象ndarray。
用于对数组执行元素级计算以及直接对数组执行数学运算的函数。
用于读写硬盘上基于数组的数据集的工具。
线性代数运算、傅里叶变换，以及随机数生成。

除了为 Python 提供快速的数组处理能力，NumPy 在数据分析方面还有另外一个主要作用，即作为在算法和库之间传递数据的容器。对于数值型数据，NumPy 数组在存储和处理数据时要比内置的 Python 数据结构高效得多。

#### pandas
pandas 提供了快速便捷处理结构化数据的大量数据结构和函数。自从 2010 年出现以来，它助使 Python 成为强大而高效的数据分析环境。本书用得最多的 pandas 对象是DataFrame，它是一个面向列（column-oriented）的二维表结构，另一个是Series，一个一维的标签化数组对象。

pandas 兼具 NumPy 高性能的数组计算功能以及电子表格和关系型数据库（如 SQL）灵活的数据处理功能。它提供了复杂精细的索引功能，能更加便捷地完成重塑、切片和切块、聚合以及选取数据子集等操作。因为数据操作、准备、清洗是数据分析最重要的技能，pandas 是本书的重点

pandas 这个名字源于面板数据（panel data，这是多维结构化数据集在计量经济学中的术语）以及 Python 数据分析（Python data analysis）。

### scipy
SciPy 是一组专门解决科学计算中各种标准问题域的包的集合，主要包括下面这些包
scipy.integrate：数值积分例程和微分方程求解器。
scipy.linalg：扩展了由numpy.linalg提供的线性代数例程和矩阵分解功能。
scipy.optimize：函数优化器（最小化器）以及根查找算法。
scipy.signal：信号处理工具。
scipy.sparse：稀疏矩阵和稀疏线性系统求解器。
scipy.special：SPECFUN（这是一个实现了许多常用数学函数（如伽玛函数）的 Fortran 库）的包装器。
scipy.stats：标准连续和离散概率分布（如密度函数、采样器、连续分布函数等）、各种统计检验方法，以及更好的描述统计法。
NumPy 和 SciPy 结合使用，便形成了一个相当完备和成熟的计算平台，可以处理多种传统的科学计算问题

### scikit-learn
分类：SVM、近邻、随机森林、逻辑回归等等。
回归：Lasso、岭回归等等。
聚类：k-均值、谱聚类等等。
降维：PCA、特征选择、矩阵分解等等。
选型：网格搜索、交叉验证、度量。
预处理：特征提取、标准化。

### statsmodels
与 scikit-learn 比较，statsmodels 包含经典统计学和经济计量学的算法。包括如下子模块：

回归模型：线性回归，广义线性模型，健壮线性模型，线性混合效应模型等等。
方差分析（ANOVA）。
时间序列分析：AR，ARMA，ARIMA，VAR 和其它模型。
非参数方法： 核密度估计，核回归。
统计模型结果可视化。

### statsmodels 更关注与统计推断，提供不确定估计和参数 p-值。相反的，scikit-learn 注重预测。

### 注意：当你使用conda和pip二者安装包时，千万不要用pip升级conda的包，这样会导致环境发生问题。当使用 Anaconda 或 Miniconda 时，最好首先使用conda进行升级

### 读者各自的工作任务不同，大体可以分为几类：

与外部世界交互

阅读编写多种文件格式和数据存储；

数据准备

清洗、修改、结合、标准化、重塑、切片、切割、转换数据，以进行分析；

转换数据

对旧的数据集进行数学和统计操作，生成新的数据集（例如，通过各组变量聚类成大的表）；

建模和计算

将数据绑定统计模型、机器学习算法、或其他计算工具；

展示

创建交互式和静态的图表可视化和文本总结。

### 引入惯例


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels as sm
```