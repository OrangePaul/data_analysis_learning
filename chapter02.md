## Python基本语法
本书大部分内容关注的是基于表格的分析和处理大规模数据集的数据准备工具。为了使用这些工具，必须首先将混乱的数据规整为整洁的表格（或结构化）形式。幸好，Python 是一个理想的语言，可以快速整理数据。Python 使用得越熟练，越容易准备新数据集以进行分析。

Python Cookbook，第 3 版，David Beazley 和 Brian K. Jones 著（O’Reilly）
流畅的 Python，Luciano Ramalho 著（O’Reilly）
高效的 Python，Brett Slatkin 著（Pearson）




```python
import numpy as np
data = {i : np.random.randn() for i in range(7)}
data
```




    {0: 0.9693269941578532,
     1: -1.259179557287676,
     2: -0.45940861250763826,
     3: 1.0803811391947753,
     4: -0.35178505657748693,
     5: 0.1775278643997852,
     6: 0.13652325907661453}




```python
%pwd
```




    '/Users/snail/Jupyter_files'




```python
def add_numbers(a, b):
    """
    Add two numbers together

    Returns
    -------
    the_sum : type of arguments
    """
    return a + b
add_numbers??
```

    Object `add_number` not found.


#### 1个？变量的话就是显示信息，函数的话显示显示函数的注释
#### 2个？？显示函数的源代码


```python
import numpy as np
np.*load*?
```

#### 可以获得包含load的numpy命名空间
np.__loader__
np.load
np.loads
np.loadtxt

### isinstance 检查对象的类型


```python
a =5
b=4.5
isinstance(a,int) # 检查a 是否是int类型
isinstance(b,(int,float))# 检查对象的类型是否在元祖中
```




    True



经常地，你可能不关心对象的类型，只关心对象是否有某些方法或用途。这通常被称为“鸭子类型”，来自“走起来像鸭子、叫起来像鸭子，那么它就是鸭子”的说法。例如，你可以通过验证一个对象是否遵循迭代协议，判断它是可迭代的。对于许多对象，这意味着它有一个__iter__魔术方法，其它更好的判断方法是使用iter函数：


```python
def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # not iterable
        return False
isiterable('a string') #True
isiterable(5) # False
```




    False



#### 用这个功能编写可以接受多种输入类型的函数。常见的例子是编写一个函数可以接受任意类型的序列（list、tuple、ndarray）或是迭代器。你可先检验对象是否是列表（或是 NUmPy 数组），如果不是的话，将其转变成列表：


```python
if not isinstance(x, list) and isiterable(x):
    x = list(x)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-21-9c2588a5d53b> in <module>
    ----> 1 if not isinstance(x, list) and isiterable(x):
          2     x = list(x)


    NameError: name 'x' is not defined


#### 要判断两个引用是否指向同一个对象，可以使用is方法。is not可以判断两个对象是不同的：
#### is 和is not 常常用来判断一个变量是否为None,因为只有一个None的实例
### 另外，None不仅是一个保留字，还是唯一的NoneType的实例


```python
a=[1,2,3]
b = a
c =list(a)
a is b #True
a is not c #True
a == c #True
```




    True




```python
a = None
a is None
```




    True



### 标量类型
None str bytes float bool int

### 数值类型
#### Python 的主要数值类型是int和float。int可以存储任意大的数
#### 浮点数使用 Python 的float类型。每个数都是双精度（64 位）的值。也可以用科学计数法表示
#### 不能得到整数的除法会得到浮点数

### 字符串
可以用单引号或双引号来写字符串，对于有换行符的字符串，可以使用三引号，'''或者"""都可以
字符串c实际包含四行文本，"""后面和lines后面的换行符。可以用count方法计算c中的新的行


```python
c = """
This is a longer string that
spans multiple lines
"""
c.count('\n') # 3
```




    3



#### 字符串格式化
字符串对象有format方法，可以替换格式化的参数为字符串，产生一个新的字符串
{0:.2f}表示格式化第一个参数为带有两位小数的浮点数。
{1:s}表示格式化第二个参数为字符串。
{2:d}表示格式化第三个参数为一个整数。


```python
template = '{0:.2f} {1:s} are worth US${2:d}'
template.format(4.55,'Pesos',1) #'4.55 Pesos are worth US$1'
```




    '4.55 Argentine Pesos are worth US$1'



### 日期和时间
#### Python 内建的datetime模块提供了datetime、date和time类型。datetime类型结合了date和time，根据datetime实例，你可以用date和time提取出各自的对象


```python
from datetime import datetime
dt = datetime(2020,3,4,16,28,59)
dt.date() # # datetime.date(2020,3,4) #都是datetime时间格式
dt.day # 4
dt.strftime('%Y-%m-%d %H:%M:%S') # '2020-03-04 16:28:59' 
# strftime方法可以将datetime格式化为字符串
#strptime可以将字符串转换成datetime对象
datetime.strptime('20091031', '%Y%m%d')
# 聚类或对时间序列进行分组，替换datetime的time字段有时会很有用
dt.replace(minute=0, second=0)
print(datetime.now())
```

    2021-03-15 16:43:20.939348



```python
# break只中断for循环的最内层，其余的for循环仍会运行：
for i in range(4):
    for j in range(4):
        if j > i:
            break
        print((i, j))
```

    (0, 0)
    (1, 0)
    (1, 1)
    (2, 0)
    (2, 1)
    (2, 2)
    (3, 0)
    (3, 1)
    (3, 2)
    (3, 3)


### pass是 Python 中的非操作语句。代码块不需要任何动作时可以使用（作为未执行代码的占位符）；因为 Python 需要使用空白字符划定代码块，所以需要pass
range函数返回一个迭代器，它产生一个均匀分布的整数序列：
range的三个参数是（起点，终点，步进）


```python

```
