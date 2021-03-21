### Pandas
##### 它含有使数据清洗和分析工作变得更快更简单的数据结构和操作工具。pandas 经常和其它工具一同使用，如数值计算工具 NumPy 和 SciPy，分析库 statsmodels 和 scikit-learn，和数据可视化库 matplotlib。pandas 是基于 NumPy 数组构建的，特别是基于数组的函数和不使用for循环的数据处理


```python
# 引入约定
import pandas as pd
# 因为Series和DataFrame用的次数非常多，所以将其引入本地命名空间中会更方便：
from pandas import Series,DataFrame
```

### Pandas 的数据结构介绍
首先就得熟悉它的两个主要数据结构：Series和DataFrame。虽然它们并不能解决所有问题，但它们为大多数应用提供了一种可靠的、易于使用的基础

### Series
Series是一种类似于一维数组的对象，它由一组数据（各种 NumPy 数据类型）以及一组与之相关的数据标签（即索引）组成。仅由一组数据即可产生最简单的Series


```python
obj=pd.Series([4,7,3,-5])
```


```python
# Series的字符串表现形式为：索引在左边，值在右边。
# 由于我们没有为数据指定索引，于是会自动创建一个 0 到N-1（N为数据的长度）的整数型索引。
# 你可以通过Series的values和index属性获取其数组表示形式和索引对象
obj.index
```




    RangeIndex(start=0, stop=4, step=1)




```python
obj.values
```




    array([ 4,  7,  3, -5])




```python
# 通常，我们希望所创建的Series带有一个可以对各个数据点进行标记的索引
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
```




    d    4
    b    7
    a   -5
    c    3
    dtype: int64




```python
# 与普通 NumPy 数组相比，你可以通过索引的方式选取Series中的单个或一组值：
obj2.index
```




    Index(['d', 'b', 'a', 'c'], dtype='object')




```python
obj2['a']
obj2[['c','a','d']]
# ['c', 'a', 'd']是索引列表，即使它包含的是字符串而不是整数
```




    c    3
    a   -5
    d    4
    dtype: int64




```python
obj2[obj2 > 0]
```




    d    4
    b    7
    c    3
    dtype: int64




```python
# 使用 NumPy 函数或类似 NumPy 的运算（如根据布尔型数组进行过滤、标量乘法、应用数学函数等）都会保留索引值的链接：
obj2*2
```




    d     8
    b    14
    a   -10
    c     6
    dtype: int64




```python
# Series看成是一个定长的有序字典，因为它是索引值到数据值的一个映射。
# 它可以用在许多原本需要字典参数的函数中：
In [24]: 'b' in obj2
Out[24]: True

In [25]: 'e' in obj2
Out[25]: False
```


```python
# 如果数据被存放在一个 Python 字典中，也可以直接通过这个字典来创建Series
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3=pd.Series(sdata)
obj3
```




    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64




```python
# 如果只传入一个字典，则结果Series中的索引就是原字典的键（有序排列）。你可以传入排好序的字典的键以改变顺序
states=['California', 'Ohio', 'Oregon', 'Texas']
obj4=pd.Series(sdata,states)
obj4
```




    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    dtype: float64




```python
# sdata中跟states索引相匹配的那 3 个值会被找出来并放到相应的位置上，
# 但由于"California"所对应的sdata值找不到，所以其结果就为NaN（即“非数字”（not a number），
# 在 pandas 中，它用于表示缺失或 NA 值）。因为Utah不在states中，它被从结果中除去

# 使用缺失（missing）或 NA 表示缺失数据。pandas 的isnull和notnull函数可用于检测缺失数据
pd.isnull(obj4)
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
pd.notnull(obj4)
```




    California    False
    Ohio           True
    Oregon         True
    Texas          True
    dtype: bool




```python
obj4.isnull()
#Series也有类似的实例方法
```




    California     True
    Ohio          False
    Oregon        False
    Texas         False
    dtype: bool




```python
# Series最重要的一个功能是，它会根据运算的索引标签自动对齐数据
In [35]: obj3
Out[35]: 
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64

In [36]: obj4
Out[36]: 
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64

In [37]: obj3 + obj4
Out[37]: 
California         NaN
Ohio           70000.0
Oregon         32000.0
Texas         142000.0
Utah               NaN
dtype: float64
```


```python
# Series对象本身及其索引都有一个name属性，该属性跟 pandas 其他的关键功能关系非常密切：
obj4.name='population'
obj4.index.name='state'
obj4
```




    state
    California        NaN
    Ohio          35000.0
    Oregon        16000.0
    Texas         71000.0
    Name: population, dtype: float64




```python
# Series的索引可以通过赋值的方式就地修改
print(obj)
obj.index=['bob','paul','jeff','ryan']
obj
```

    0    4
    1    7
    2    3
    3   -5
    dtype: int64





    bob     4
    paul    7
    jeff    3
    ryan   -5
    dtype: int64



### DataFrame
##### DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。DataFrame既有行索引也有列索引，它可以被看做由Series组成的字典（共用同一个索引）。DataFrame中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）。有关DataFrame内部的技术细节远远超出了本书所讨论的范围。



```python
# 建DataFrame的办法有很多，最常用的一种是直接传入一个由等长列表或 NumPy 数组组成的字典：
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
frame
# 结果DataFrame会自动加上索引（跟Series一样），且全部列会被有序排列
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nevada</td>
      <td>2003</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对于特别大的DataFrame，head方法会选取前五行
frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果指定了列序列，则DataFrame的列就会按照指定顺序进行排列
pd.DataFrame(data,columns=['year','state','pop'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果传入的列在数据中找不到，就会在结果集中产生缺失值
frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four',
                             'five', 'six'])
frame2
frame2.columns
```




    Index(['year', 'state', 'pop', 'debt'], dtype='object')




```python
# 通过类似字典标记的方式或属性的方式，可以将DataFrame的列获取为一个Series
frame2['state']
# 等价
frame2.state
```




    one        Ohio
    two        Ohio
    three      Ohio
    four     Nevada
    five     Nevada
    six      Nevada
    Name: state, dtype: object




```python
frame2.loc['three']
#行也可以通过位置或名称的方式进行获取，比如用loc属性
```




    year     2002
    state    Ohio
    pop       3.6
    debt      NaN
    Name: three, dtype: object




```python
# 列可以通过赋值的方式进行修改。例如，我们可以给那个空的"debt"列赋上一个标量值或一组值
import numpy as np
frame2['debt']=16.5 # 默认所有值都是这个
frame2['debt']=np.arange(1.,7.)
frame2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将列表或数组赋值给某个列时，其长度必须跟DataFrame的长度相匹配。
# 如果赋值的是一个Series，就会精确匹配DataFrame的索引，所有的空位都将被填上缺失值
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['dbet']=val
```


```python
# 为不存在的列赋值会创建出一个新列
frame2['west'] = frame2.state=='Nevada' # 不能frame2.west来创建
frame2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
      <th>dbet</th>
      <th>west</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>2.0</td>
      <td>-1.2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>4.0</td>
      <td>-1.5</td>
      <td>True</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>5.0</td>
      <td>-1.7</td>
      <td>True</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# del方法可以删除列
del frame2['west']
frame2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
      <th>dbet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>2.0</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>four</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>4.0</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>five</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>5.0</td>
      <td>-1.7</td>
    </tr>
    <tr>
      <th>six</th>
      <td>2003</td>
      <td>Nevada</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 通过索引方式返回的列只是相应数据的视图而已，并不是副本。
#### 因此，对返回的Series所做的任何就地修改全都会反映到源DataFrame上。
#### 通过Series的copy方法即可指定复制列


```python
# 另一种常见的数据形式是嵌套字典
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3=pd.DataFrame(pop)
# 如果嵌套字典传给DataFrame，pandas 就会被解释为：外层字典的键作为列，内层键则作为行索引
frame3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 可以使用类似Numpy数组的方法，对DataFrame进行转置
frame3.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2001</th>
      <th>2002</th>
      <th>2000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nevada</th>
      <td>2.4</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>1.7</td>
      <td>3.6</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 内层字典的键会被合并、排序以形成最终的索引。如果明确指定了索引，则不会这样
pd.DataFrame(pop,index=[2001,2002,2003])

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果设置了DataFrame的index和columns的name属性
frame3.index.name = 'year'; frame3.columns.name = 'state' # 如下图，看着挺别扭
frame3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>state</th>
      <th>Nevada</th>
      <th>Ohio</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001</th>
      <td>2.4</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>2.9</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>NaN</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 和Series一样，values属性也会以二维ndarray的形势返回DF中的数据
frame3.values
```




    array([[2.4, 1.7],
           [2.9, 3.6],
           [nan, 1.5]])




```python
frame2.values
# 如果DataFrame各列的数据类型不同，则值数组的dtype就会选用能兼容所有列的数据类型
```




    array([[2000, 'Ohio', 1.5, 1.0, nan],
           [2001, 'Ohio', 1.7, 2.0, -1.2],
           [2002, 'Ohio', 3.6, 3.0, nan],
           [2001, 'Nevada', 2.4, 4.0, -1.5],
           [2002, 'Nevada', 2.9, 5.0, -1.7],
           [2003, 'Nevada', 3.2, 6.0, nan]], dtype=object)



### 索引对象
#### pandas 的索引对象负责管理轴标签和其他元数据（比如轴名称等）。构建Series或DataFrame时，所用到的任何数组或其他序列的标签都会被转换成一个Index
#### 特点 不可变，可以重复


```python
obj = pd.Series(range(3), index=['a', 'b', 'c'])
index=obj.index
index
```




    Index(['a', 'b', 'c'], dtype='object')




```python
# Index对象是不可变的，因此用户不能对其进行修改
index[1]='d'
# 不可变可以使Index对象在多个数据结构之间安全共享
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-33-80c04a3e64e9> in <module>
          1 # Index对象是不可变的，因此用户不能对其进行修改
    ----> 2 index[1]='d'
    

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py in __setitem__(self, key, value)
       4079 
       4080     def __setitem__(self, key, value):
    -> 4081         raise TypeError("Index does not support mutable operations")
       4082 
       4083     def __getitem__(self, key):


    TypeError: Index does not support mutable operations



```python
labels = pd.Index(np.arange(3))
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2.index is labels
```




    True




```python
frame3.index
```




    Int64Index([2001, 2002, 2000], dtype='int64', name='year')




```python
2003 in frame3.index
```




    False




```python
# 与 python 的集合不同，pandas 的Index可以包含重复的标签
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])
dup_labels
```




    Index(['foo', 'foo', 'bar', 'bar'], dtype='object')



### 基本功能

####  重新索引


```python
# pandas 对象的一个重要方法是reindex，其作用是创建一个新对象，它的数据符合新的索引。
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

```


```python
#用该Series的reindex将会根据新索引进行重排。如果某个索引值当前不存在，就引入缺失值
obj2=obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
```




    a   -5.3
    b    7.2
    c    3.6
    d    4.5
    e    NaN
    dtype: float64




```python
# 对于时间序列这样的有序数据，重新索引时可能需要做一些插值处理。
# method选项即可达到此目的，例如，使用ffill可以实现前向值填充：
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3
```




    0      blue
    2    purple
    4    yellow
    dtype: object




```python
obj3.reindex(range(6), method='ffill')
```




    0      blue
    1      blue
    2    purple
    3    purple
    4    yellow
    5    yellow
    dtype: object




```python
# 借助DataFrame，reindex可以修改（行）索引和列。只传递一个序列时，会重新索引结果的行
fram = pd.DataFrame(np.arange(9).reshape((3, 3)),
                 index=['a', 'c', 'd'],
                columns=['Ohio', 'Texas', 'California'])
fram
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
fram2=fram.reindex(['a', 'b', 'c', 'd'])
fram2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ohio</th>
      <th>Texas</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 列可以用columns关键字重新索引
states = ['Texas', 'Utah', 'California']
fram2.reindex(columns=states)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Texas</th>
      <th>Utah</th>
      <th>California</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 丢弃某条轴上的一个或多个项很简单，只要有一个索引数组或列表即可。
# 由于需要执行一些数据整理和集合逻辑，所以drop方法返回的是一个在指定轴上删除了指定值的新对象
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj
```




    a    0.0
    b    1.0
    c    2.0
    d    3.0
    e    4.0
    dtype: float64




```python
new_obj=obj.drop('c') # 原来的obj并没有改变
new_obj
```




    a    0.0
    b    1.0
    d    3.0
    e    4.0
    dtype: float64




```python
# 对于DataFrame，可以删除任意轴上的索引值。
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['Utah','Ohio']) # data的值没有变
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 通过传递axis=1或axis='columns' 可以删除列的值
data.drop('two',axis=1,inplace=True) # 用inplace是真的删除原来的数据，data值变化了
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



### 索引 选取  过滤
Series索引（obj[...]）的工作方式类似于 NumPy 数组的索引，只不过Series的索引值不只是整数


```python
obj = pd.Series(np.arange(1.,5.), index=['a', 'b', 'c', 'd'])
obj
```




    a    1.0
    b    2.0
    c    3.0
    d    4.0
    dtype: float64




```python
print(obj[0])
obj['a']
```

    0.0





    0.0




```python
obj[['a','b']]
```




    a    1.0
    b    5.0
    dtype: float64




```python
obj[[1,3]]
```




    b    2.0
    d    4.0
    dtype: float64




```python
obj[obj>1]
```




    b    2.0
    c    3.0
    d    4.0
    dtype: float64




```python
# p.s 利用标签的切片运算与普通的 Python 切片运算不同，其末端是包含的：
obj['b':'c']
```




    b    2.0
    c    3.0
    dtype: float64




```python
# 用切片可以对Series的相应部分进行设置
obj['b':'c'] = 5 # 会修改本体原来的内容
obj
```




    a    1.0
    b    5.0
    c    5.0
    d    4.0
    dtype: float64



##### DataFrame索引
用一个值或序列对DataFrame进行索引其实就是获取一个或多个列


```python
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['one']
data[['one','four','three']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>four</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>15</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['Utah']# 不能这样选取行
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2894             try:
    -> 2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'Utah'

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-90-a8eed22065c0> in <module>
    ----> 1 data['Utah']
    

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2900             if self.columns.nlevels > 1:
       2901                 return self._getitem_multilevel(key)
    -> 2902             indexer = self.columns.get_loc(key)
       2903             if is_integer(indexer):
       2904                 indexer = [indexer]


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    -> 2897                 raise KeyError(key) from err
       2898 
       2899         if tolerance is not None:


    KeyError: 'Utah'



```python
data[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data['four']>5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
#另一种就是布尔型进行索引
data<5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[data<5]=0
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ohio</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



### 用loc和iloc进行选取
对于DataFrame的行的标签索引，我引入了特殊的标签运算符loc和iloc。它们可以让你用类似 NumPy 的标记，使用轴标签（loc）或整数索引（iloc），从DataFrame选择行和列的子集



```python
data.loc['Colorado', ['two', 'three']] # 轴标签
```




    two      5
    three    6
    Name: Colorado, dtype: int64




```python
# 然后用iloc和整数进行选取：
data.iloc[2, [3, 0, 1]]# 整数索引
```




    four    11
    one      8
    two      9
    Name: Utah, dtype: int64




```python
data.iloc[2]
```




    one       8
    two       9
    three    10
    four     11
    Name: Utah, dtype: int64




```python
data.iloc[[1, 2], [3, 0, 1]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>four</th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>7</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>11</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 这两个索引函数也适用于一个标签或多个标签的切片
data.loc[:'Utah','two']
```




    Ohio        0
    Colorado    5
    Utah        9
    Name: two, dtype: int64




```python
data.iloc[:3, :3][data.three>5]
```

    <ipython-input-116-cd9f8d8ee192>:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      data.iloc[:3, :3][data.three>5]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>0</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### 整数索引
处理整数索引的 pandas 对象常常难住新手，因为它与 Python 内置的列表和元组的索引语法不同
#### 为了进行统一，如果轴索引含有整数，数据选取总会使用标签。为了更准确，请使用loc（标签）或iloc（整数）


```python
ser = pd.Series(np.arange(3.))
ser
```




    0    0.0
    1    1.0
    2    2.0
    dtype: float64




```python
ser[-1]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/range.py in get_loc(self, key, method, tolerance)
        354                 try:
    --> 355                     return self._range.index(new_key)
        356                 except ValueError as err:


    ValueError: -1 is not in range

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-119-44969a759c20> in <module>
    ----> 1 ser[-1]
    

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/series.py in __getitem__(self, key)
        880 
        881         elif key_is_scalar:
    --> 882             return self._get_value(key)
        883 
        884         if is_hashable(key):


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/series.py in _get_value(self, label, takeable)
        987 
        988         # Similar to Index.get_value, but we do not fall back to positional
    --> 989         loc = self.index.get_loc(label)
        990         return self.index._get_values_for_loc(self, loc, label)
        991 


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/range.py in get_loc(self, key, method, tolerance)
        355                     return self._range.index(new_key)
        356                 except ValueError as err:
    --> 357                     raise KeyError(key) from err
        358             raise KeyError(key)
        359         return super().get_loc(key, method=method, tolerance=tolerance)


    KeyError: -1



```python
ser.loc[:1] #location 左闭右闭
```




    0    0.0
    1    1.0
    dtype: float64




```python
ser.iloc[:1] # index  location 左闭右开
```




    0    0.0
    dtype: float64



### 算术运算和数据对齐
#### pandas 最重要的一个功能是，它可以对不同索引的对象进行算术运算。在将对象相加时，如果存在不同的索引对，则结果的索引就是该索引对的并集。


```python
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
```


```python
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1],
           index=['a', 'c', 'e', 'f', 'g'])
```


```python
s1+s2
```




    a    5.2
    c    1.1
    d    NaN
    e    0.0
    f    NaN
    g    NaN
    dtype: float64



#### 对于DataFrame


```python
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                   index=['Ohio', 'Texas', 'Colorado'])
```


```python
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
```


```python
df1+df2
# 自动的数据对齐操作在不重叠的索引处引入了 NA 值。缺失值会在算术运算过程中传播
# 如果DataFrame对象相加，没有共用的列或行标签，结果都会是空
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Colorado</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                    columns=list('abcd'))
df4 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                    columns=list('abcde'))
```


```python
# 在对不同索引的对象进行算术运算时，你可能希望当一个对象中某个轴标签在另一个对象中找不到时填充一个特殊值
df4.loc[1,'b']=np.nan
df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3+df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用df1的add方法，传入df2以及一个fill_value参数
df3.add(df4,fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#1/df3# 等价于
df3.rdiv(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>inf</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.250</td>
      <td>0.200000</td>
      <td>0.166667</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.125</td>
      <td>0.111111</td>
      <td>0.100000</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 在对Series或DataFrame重新索引时，也可以指定一个填充值
df3.reindex(columns=df4.columns,fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame和Series之间的运算
跟不同维度的 NumPy 数组一样，DataFrame和Series之间算术运算也是有明确规定的。先来看一个具有启发性的例子，计算一个二维数组与其某行之间的差


```python
In [175]: arr = np.arange(12.).reshape((3, 4))

In [176]: arr
Out[176]: 
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])

In [177]: arr[0]
Out[177]: array([ 0.,  1.,  2.,  3.])

In [178]: arr - arr[0]
Out[178]: 
array([[ 0.,  0.,  0.,  0.],
       [ 4.,  4.,  4.,  4.],
       [ 8.,  8.,  8.,  8.]])
# 当我们从arr减去arr[0]，每一行都会执行这个操作。这就叫做广播（broadcasting）
```


```python
 frame = pd.DataFrame(np.arange(12.).reshape((4, 3)),
                      columns=list('bde'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])

series = frame.iloc[0]
series
```




    b    0.0
    d    1.0
    e    2.0
    Name: Utah, dtype: float64




```python
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 默认情况下，DataFrame和Series之间的算术运算会将Series的索引匹配到DataFrame的列.然后一直向下广播

```


```python
frame-series
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>9.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
series2 = pd.Series(range(3), index=['b', 'e', 'f'])
series2
```




    b    0
    e    1
    f    2
    dtype: int64




```python
frame+series2
# 某个索引值在DataFrame的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
      <th>f</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>6.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
series3=frame['d']
```


```python
series3
```




    Utah       1.0
    Ohio       4.0
    Texas      7.0
    Oregon    10.0
    Name: d, dtype: float64




```python
# 如果要匹配且在列上广播，则必须用算数方法
frame.sub(series3,axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 函数应用和映射


```python
# NumPy 的ufuncs（元素级数组方法）也可用于操作 pandas 对象
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),
                      index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>1.130842</td>
      <td>1.997379</td>
      <td>0.298605</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>0.732091</td>
      <td>0.065512</td>
      <td>1.317292</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.354221</td>
      <td>1.333507</td>
      <td>0.352927</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>0.417321</td>
      <td>0.888456</td>
      <td>2.039939</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 另一个常见的操作是，将函数应用到由各列或行所形成的一维数组上。DataFrame的apply方法即可实现此功能
f= lambda x:x.max()-x.min()
frame.apply(f) #默认对列应用函数
```




    b    1.862934
    d    3.330886
    e    2.392867
    dtype: float64




```python
# 这里的函数f，计算了一个Series的最大值和最小值的差，在frame的每列都执行了一次。结果是一个Series，使用frame的列作为索引
# 如果传递axis='columns'到apply，这个函数会在每行执行
frame.apply(f,axis=1)
```




    Utah      3.128221
    Ohio      2.049383
    Texas     1.687729
    Oregon    2.928395
    dtype: float64




```python
# 许多最为常见的数组统计功能都被实现成DataFrame的方法（如sum和mean），因此无需使用apply方法
# 传递到apply的函数不是必须返回一个标量，还可以返回由多个值组成的Series
def ff(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])
frame.apply(ff)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>-0.732091</td>
      <td>-1.997379</td>
      <td>-0.352927</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.130842</td>
      <td>1.333507</td>
      <td>2.039939</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 元素级的 Python 函数也是可以用的。假如你想得到frame中各个浮点值的格式化字符串，使用applymap即可
format = lambda x: '%.2f' % x
frame.applymap(format)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Utah</th>
      <td>1.13</td>
      <td>-2.00</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-0.73</td>
      <td>-0.07</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>-0.35</td>
      <td>1.33</td>
      <td>-0.35</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>-0.42</td>
      <td>-0.89</td>
      <td>2.04</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 所以叫做applymap，是因为Series有一个用于应用元素级函数的map方法
In [200]: frame['e'].map(format)
Out[200]: 
Utah      -0.52
Ohio       1.39
Texas      0.77
Oregon    -1.30
Name: e, dtype: object
```

### 排序和排名
根据条件对数据集排序（sorting）也是一种重要的内置运算。要对行或列索引进行排序（按字典顺序），可使用sort_index方法，它将返回一个已排序的新对象


```python
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
```




    a    1
    b    2
    c    3
    d    0
    dtype: int64




```python
# DataFrame，则可以根据任意一个轴上的索引进行排序
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                    index=['three', 'one'],
                     columns=['d', 'a', 'b', 'c'])
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_index(axis=1,ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>three</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>one</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 若要按值对Series进行排序，可使用其sort_values方法
obj = pd.Series([4, 7,np.nan, -3, 2])
obj.sort_values(ascending=False) # 任何缺失值都会放到末尾
```




    1    7.0
    0    4.0
    4    2.0
    3   -3.0
    2    NaN
    dtype: float64




```python
# 排序一个DataFrame时，你可能希望根据一个或多个列中的值进行排序。
# 将一个或多个列的名字传递给sort_values的by选项即可达到该目的
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.sort_values(by=['a','b']) # 根据多个列进行排序，传入名称的列表即可
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 排名会从 1 开始一直到数组中有效数据的数量。接下来介绍Series和DataFrame的rank方法。
# 默认情况下，rank是通过“为各组分配一个平均排名”的方式破坏平级关系的
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()
```




    0    6.5
    1    1.0
    2    6.5
    3    4.5
    4    3.0
    5    2.0
    6    4.5
    dtype: float64




```python
# 也可以根据值在原数据中出现的顺序给出排名：
obj.rank(method='first') #条目 0 和 2 没有使用平均排名 6.5，它们被设成了 6 和 7，因为数据中标签 0 位于标签 2 的前面
```




    0    6.0
    1    1.0
    2    7.0
    3    4.0
    4    3.0
    5    2.0
    6    5.0
    dtype: float64




```python
# DataFrame可以在行或列上计算排名
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                       'c': [-2, 5, 8, -2.5]})
frame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.3</td>
      <td>0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.0</td>
      <td>0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1</td>
      <td>-2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame.rank(axis='columns') #按照列的大小排名
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>a</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 带有重复标签的轴索引
介绍的所有范例都有着唯一的轴标签（索引值）。虽然许多 pandas 函数（如reindex）都要求标签唯一，但这并不是强制性的。我们来看看下面这个简单的带有重复索引值的Series


```python
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique
```




    False




```python
# 对于带有重复值的索引，数据选取的行为将会有些不同。
# 如果某个索引对应多个值，则返回一个Series；而对应单个值的，则返回一个标量值
In [225]: obj['a']
Out[225]: 
a    0
a    1
dtype: int64

In [226]: obj['c']
Out[226]: 4
```

### 汇总和计算描述统计
pandas 对象拥有一组常用的数学和统计方法。它们大部分都属于约简和汇总统计，

用于从Series中提取单个值（如sum或mean）或从DataFrame的行或列中提取一个Series。

跟对应的 NumPy 数组方法相比，它们都是基于没有缺失数据的假设而构建的。看一个简单的DataFrame


```python
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                    [np.nan, np.nan], [0.75, -1.3]],
                   index=['a', 'b', 'c', 'd'],
                   columns=['one', 'two'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.10</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.75</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sum() # 默认对列
```




    one    9.25
    two   -5.80
    dtype: float64




```python
df.sum(axis=1) # 对行求和运算
# df.sum(axis=1,skipna=False)
# NA 值会自动被排除，除非整个切片（这里指的是行或列）都是 NA。通过skipna选项可以禁用该功能
```




    a    1.40
    b    2.60
    c    0.00
    d   -0.55
    dtype: float64




```python
# （如idxmin和idxmax）返回的是间接统计（比如达到最小值或最大值的索引）
df.idxmax()
```




    one    b
    two    d
    dtype: object




```python
# 累加
df.cumsum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>8.50</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>9.25</td>
      <td>-5.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 还有一种方法，它既不是约简型也不是累计型。describe就是一个例子，它用于一次性产生多个汇总统计
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.083333</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.493685</td>
      <td>2.262742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.750000</td>
      <td>-4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.075000</td>
      <td>-3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.250000</td>
      <td>-2.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.100000</td>
      <td>-1.300000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 还有一种方法，它既不是约简型也不是累计型。describe就是一个例子，它用于一次性产生多个汇总统计
obj=pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
```




    count     16
    unique     3
    top        a
    freq       8
    dtype: object



### 唯一值、值计数以及成员资格


```python
# 还有一类方法可以从一维Series的值中抽取信息
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
# unique，它可以得到Series中的唯一值数组
unique=obj.unique()
unique
```




    array(['c', 'a', 'd', 'b'], dtype=object)




```python
# value_counts用于计算一个Series中各值出现的频率
obj.value_counts()# 结果Series是按值频率降序排列的
```




    a    3
    c    3
    b    2
    d    1
    dtype: int64




```python
# value_counts还是一个顶级 pandas 方法，可用于任何数组或序列
pd.value_counts(obj.values,sort=False)
```




    b    2
    c    3
    a    3
    d    1
    dtype: int64




```python
# isin用于判断向量化集合的成员资格，可用于过滤Series中或DataFrame列中数据的子集
mask=obj.isin(['b','c'])
mask
```




    0     True
    1    False
    2    False
    3    False
    4    False
    5     True
    6     True
    7     True
    8     True
    dtype: bool




```python
obj[mask]
```




    0    c
    5    b
    6    b
    7    c
    8    c
    dtype: object




```python
# 与isin类似的是Index.get_indexer方法，它可以给你一个索引数组，从可能包含重复值的数组到另一个不同值的数组
In [260]: to_match = pd.Series(['c', 'a', 'b', 'b', 'c', 'a'])

In [261]: unique_vals = pd.Series(['c', 'b', 'a'])

In [262]: pd.Index(unique_vals).get_indexer(to_match)
Out[262]: array([0, 2, 1, 1, 0, 2])
```


```python
# 你可能希望得到DataFrame中多个相关列的一张柱状图
data = pd.DataFrame({'Qu1': [1, 3, 4, 3, 4],
                      'Qu2': [2, 3, 1, 2, 3],
                     'Qu3': [1, 5, 2, 4, 4]})
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#pandas.value_counts传给该DataFrame的apply函数，就会出现
result = data.apply(pd.value_counts).fillna(0)
result
# 行标签 12345是各种不同的值
# 对应的值是在某列出现的次数
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Qu1</th>
      <th>Qu2</th>
      <th>Qu3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
