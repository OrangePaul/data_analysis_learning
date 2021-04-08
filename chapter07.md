### 处理缺失数据


```python
# pandas对象的所有描述性统计默认都包括缺失数据
import pandas as pd
ss=pd.read_csv('deepshare/iris.csv')
ss.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 6 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   ID            150 non-null    int64  
     1   Sepal.Length  150 non-null    float64
     2   Sepal.Width   150 non-null    float64
     3   Petal.Length  150 non-null    float64
     4   Petal.Width   150 non-null    float64
     5   Species       150 non-null    object 
    dtypes: float64(4), int64(1), object(1)
    memory usage: 7.2+ KB



```python
# pandas 使用浮点值NAN表示缺失数据，也叫哨兵值
import numpy as np
str_data=pd.Series(['aard','artich',np.nan,'avocado'])
str_data
```




    0       aard
    1     artich
    2        NaN
    3    avocado
    dtype: object




```python
# python内置的None值在对象数组中也可以作为NA
str_data[0]=None
str_data.isnull()
```




    0     True
    1    False
    2     True
    3    False
    dtype: bool



dropna 根据各标签的值中是否存在缺失数据对轴标签进行过滤

fillna 用指定值或插值方法填充缺失数据

isnull 返回一个含有布尔值的对象，表示哪些值是缺失值/na

notnull isnull的否定式

#### 滤除缺失数据


```python
# 对一个Series，dropna返回一个仅含非空数据和索引的Series
from numpy import nan as NA
data = pd.Series([1,NA,3.5,NA,7])
data.dropna()
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
# 等价于
data[data.notnull()]
```




    0    1.0
    2    3.5
    4    7.0
    dtype: float64




```python
# 对于DF对象，可能希望丢弃全NA或者含有NA的行或者列。 dropna默认丢弃任何含有缺失值的行
data= pd.DataFrame([[1,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.6,3]])
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.6</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
cleaned=data.dropna()
cleaned
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 传入how=all将只丢弃全为NA的行
data.dropna(how='all')
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.6</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[3]=NA
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.6</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 丢弃列，需要多传入axis=1
data.dropna(axis=1,how='all')
# 原来的data内容不变
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>6.6</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df.dropna(thresh=n)
# 这一行出去NA值，剩余数值的数量>=n,就显示该行
df=pd.DataFrame(np.random.randn(7,3))
df.iloc[:4,1]=NA
df.iloc[:2,2]=NA
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.393062</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.109624</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.789360</td>
      <td>NaN</td>
      <td>-0.490531</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016961</td>
      <td>NaN</td>
      <td>-0.533119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.359567</td>
      <td>-0.696272</td>
      <td>0.484617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.354631</td>
      <td>0.940333</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.470994</td>
      <td>1.482785</td>
      <td>-2.279891</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(thresh=2)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.789360</td>
      <td>NaN</td>
      <td>-0.490531</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016961</td>
      <td>NaN</td>
      <td>-0.533119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.359567</td>
      <td>-0.696272</td>
      <td>0.484617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.354631</td>
      <td>0.940333</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.470994</td>
      <td>1.482785</td>
      <td>-2.279891</td>
    </tr>
  </tbody>
</table>
</div>



#### 填充缺失数据


```python
# 一些场景下，可能不想滤除缺失数据（可能会丢弃跟它有关的其他数据）
# 通过填补，就可以用到fillna
df.fillna(0)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.393062</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.109624</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.789360</td>
      <td>0.000000</td>
      <td>-0.490531</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016961</td>
      <td>0.000000</td>
      <td>-0.533119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.359567</td>
      <td>-0.696272</td>
      <td>0.484617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.354631</td>
      <td>0.940333</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.470994</td>
      <td>1.482785</td>
      <td>-2.279891</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 通过字典调用，可以实现不同列填充不同的值
df.fillna({1:0.8,2:0})
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.393062</td>
      <td>0.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.109624</td>
      <td>0.800000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.789360</td>
      <td>0.800000</td>
      <td>-0.490531</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016961</td>
      <td>0.800000</td>
      <td>-0.533119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.359567</td>
      <td>-0.696272</td>
      <td>0.484617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.354631</td>
      <td>0.940333</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.470994</td>
      <td>1.482785</td>
      <td>-2.279891</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fillna 默认会返回新对象，也可以设置inplace=True对现有对象进行就地修改
_=df.fillna(0,inplace=True) # 修改调用者对象而不产生副本
# fillna中method=ffill插值的方法可以向下填充 limit表示可以连续填充的最大数量
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.393062</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.109624</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.789360</td>
      <td>0.000000</td>
      <td>-0.490531</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.016961</td>
      <td>0.000000</td>
      <td>-0.533119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.359567</td>
      <td>-0.696272</td>
      <td>0.484617</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.354631</td>
      <td>0.940333</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.470994</td>
      <td>1.482785</td>
      <td>-2.279891</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 另外可以传入Series的平均值或者中位数
data=pd.Series([1.,NA,3.5,NA,7])
data.fillna(data.mean())
```




    0    1.000000
    1    3.833333
    2    3.500000
    3    3.833333
    4    7.000000
    dtype: float64



### 数据转换

#### 移除重复数据


```python
# DF中出现重复行有多个原因
data = pd.DataFrame({'k1':['one','two']* 3 +['two'],'k2':[1,1,2,3,3,4,4]})
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# duplicated方法返回一个布尔型Series 表示各行是否重复
data.duplicated()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    5    False
    6     True
    dtype: bool




```python
data.drop_duplicates()
#返回DF
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
      <th>k1</th>
      <th>k2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>one</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>two</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>two</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 以上2种方法默认会判断全部列，也可以指定部分列进行重复项判断
data['v1']=range(7)
data.drop_duplicates(['k1'])
# duplicated，drop_duplicates默认保留的是第一个出现的值组合，传入keep=’last‘ 则保留最后一个
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
      <th>k1</th>
      <th>k2</th>
      <th>v1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### 利用函数或者映射进行数据转换



```python
# 有时候希望根据数组，Series,DF列中的值来实现转换工作
data=pd.DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef',
                           'Bacon','pastrami','honey ham','nova lox'],
                  'ounces':[4,3,12,6,7.6,8,3,5,6]})
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
      <th>food</th>
      <th>ounces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#假设要添加一列表示该食物来源的动物类型
meat_to_animal={'bacon':'pig','pulled pork':'pig',
               'pastrami':'cow','corned beef':'cow',
               'honey ham':'pig','nova lox':'salmon'}
```


```python
# 先用str.lower 各值转换成小写
lowercase=data['food'].str.lower()
lowercase
```




    0          bacon
    1    pulled pork
    2          bacon
    3       pastrami
    4    corned beef
    5          bacon
    6       pastrami
    7      honey ham
    8       nova lox
    Name: food, dtype: object




```python
data['animal']=lowercase.map(meat_to_animal)
# 可以传入一个函数来处理
# data['food'].map(lambda x:meat_toanimal[x.lower()])
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
      <th>food</th>
      <th>ounces</th>
      <th>animal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bacon</td>
      <td>4.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pulled pork</td>
      <td>3.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bacon</td>
      <td>12.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pastrami</td>
      <td>6.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>4</th>
      <td>corned beef</td>
      <td>7.6</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bacon</td>
      <td>8.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pastrami</td>
      <td>3.0</td>
      <td>cow</td>
    </tr>
    <tr>
      <th>7</th>
      <td>honey ham</td>
      <td>5.0</td>
      <td>pig</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nova lox</td>
      <td>6.0</td>
      <td>salmon</td>
    </tr>
  </tbody>
</table>
</div>



#### 替换值


```python
# 利用fillna填充可以看做值替换的一种特殊情况，map可以修改对象的数据子集，
# replace 提供了一种实现该功能更灵活的方式
data= pd.Series([1.,-999.,2.,-999.,-1000.,3.])
data
```




    0       1.0
    1    -999.0
    2       2.0
    3    -999.0
    4   -1000.0
    5       3.0
    dtype: float64




```python
# -999可能是一个表示缺失的标记值，将其替换为pandas能够理解的NA值，可以利用replace
# 来产生一个新的Series(除非传入inplace=True)
data.replace(-999,np.nan)
```




    0       1.0
    1       NaN
    2       2.0
    3       NaN
    4   -1000.0
    5       3.0
    dtype: float64




```python
# 若果希望一次性替换多个值，可以出啊如一个由待替换值组成的列表一级一个替换值
data.replace([-999,-1000],np.nan)
# 每个值有不同替换值，可以传递一个替换列表
data.replace([-999,-1000],[np.nan,0]) # 这里传入参数可以是字典
```




    0    1.0
    1    NaN
    2    2.0
    3    NaN
    4    0.0
    5    3.0
    dtype: float64



data.replace方法与data.str.replace不同，后者做的是字符串的元素及替换

#### 重命名轴索引


```python
#跟Series中的值一样，轴标签也可以通过函数或者映射转换
data = pd.DataFrame(np.arange(12).reshape((3, 4)),
                     index=['Ohio', 'Colorado', 'New York'],
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
      <th>New York</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
#跟Series一样，轴索引
transform = lambda x: x[:4].upper()
data.index.map(transform)
```




    Index(['OHIO', 'COLO', 'NEW '], dtype='object')




```python
#将其赋值给index，这样就可以对DataFrame进行就地修改
data.index = data.index.map(transform)
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
      <th>OHIO</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
#如果想要创建数据集的转换版（而不是修改原始数据），比较实用的方法是rename
data.rename(index=str.title, columns=str.upper)
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
      <th>ONE</th>
      <th>TWO</th>
      <th>THREE</th>
      <th>FOUR</th>
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
      <th>Colo</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>New</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 特别说明一下，rename可以结合字典型对象实现对部分轴标签的更新：
data.rename(index={'OHIO': 'INDIANA'},
    columns={'three': 'peekaboo'})
# rename可以实现复制DataFrame并对其索引和列标签进行赋值。如果希望就地修改某个数据集，传入inplace=True即可
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
      <th>peekaboo</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INDIANA</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>COLO</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>NEW</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



#### 离散化和面元划分

为了便于分析，连续数据常常被离散化或拆分为“面元”（bin）。假设有一组人员数据，而你希望将它们划分为不同的年龄组：

接下来将这些数据划分为“18 到 25”、“26 到 35”、“35 到 60”以及“60 以上”几个面元。要实现该功能，你需要使用 pandas 的cut函数


```python
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins=[19,25,35,60,100]
cats=pd.cut(ages,bins) # 圆括号表示开端，而方括号则表示闭端（包括）。哪边是闭端可以通过right=False进行修改
```


```python
cats
```




    [(19, 25], (19, 25], (19, 25], (25, 35], (19, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
    Length: 12
    Categories (4, interval[int64]): [(19, 25] < (25, 35] < (35, 60] < (60, 100]]




```python
cats.codes
# pandas 返回的是一个特殊的Categorical对象。结果展示了pandas.cut划分的面元。你可以将其看做一组表示面元名称的字符串。
# 它的底层含有一个表示不同分类名称的类型数组，以及一个codes属性中的年龄数据的标签
```




    array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)




```python
cats.categories
```




    IntervalIndex([(19, 25], (25, 35], (35, 60], (60, 100]],
                  closed='right',
                  dtype='interval[int64]')




```python
pd.value_counts(cats)
# pd.value_counts(cats)是pandas.cut结果的面元计数
```




    (19, 25]     5
    (35, 60]     3
    (25, 35]     3
    (60, 100]    1
    dtype: int64




```python
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages,bins,labels=group_names)
#可以通过传递一个列表或数组到labels，设置自己的面元名称
```




    ['Youth', 'Youth', 'Youth', 'YoungAdult', 'Youth', ..., 'YoungAdult', 'Senior', 'MiddleAged', 'MiddleAged', 'YoungAdult']
    Length: 12
    Categories (4, object): ['Youth' < 'YoungAdult' < 'MiddleAged' < 'Senior']




```python
# 如果向cut传入的是面元的数量而不是确切的面元边界，则它会根据数据的最小值和最大值计算等长面元。
data=np.random.rand(20)
data
```




    array([0.64671347, 0.55164183, 0.20901415, 0.60458035, 0.96336985,
           0.65095625, 0.53263937, 0.49541416, 0.9715133 , 0.69191081,
           0.95174213, 0.69449069, 0.40145349, 0.54015644, 0.6202431 ,
           0.99337942, 0.21061173, 0.34532927, 0.60954078, 0.93119453])




```python
pd.cut(data,4,precision=2)
```




    [(0.6, 0.8], (0.41, 0.6], (0.21, 0.41], (0.6, 0.8], (0.8, 0.99], ..., (0.8, 0.99], (0.21, 0.41], (0.21, 0.41], (0.6, 0.8], (0.8, 0.99]]
    Length: 20
    Categories (4, interval[float64]): [(0.21, 0.41] < (0.41, 0.6] < (0.6, 0.8] < (0.8, 0.99]]




```python
# qcut是一个非常类似于cut的函数，它可以根据样本分位数对数据进行面元划分。
# 根据数据的分布情况，cut可能无法使各个面元中含有相同数量的数据点。\
# 而qcut由于使用的是样本分位数，因此可以得到大小基本相等的面元
data = np.random.randn(1000)
cats=pd.qcut(data,4)
cats
```




    [(-3.331, -0.67], (0.641, 3.305], (0.641, 3.305], (-0.67, -0.0178], (-0.67, -0.0178], ..., (-0.67, -0.0178], (-0.0178, 0.641], (0.641, 3.305], (0.641, 3.305], (0.641, 3.305]]
    Length: 1000
    Categories (4, interval[float64]): [(-3.331, -0.67] < (-0.67, -0.0178] < (-0.0178, 0.641] < (0.641, 3.305]]




```python
pd.value_counts(cats)
```




    (0.641, 3.305]      250
    (-0.0178, 0.641]    250
    (-0.67, -0.0178]    250
    (-3.331, -0.67]     250
    dtype: int64




```python
# 与cut类似，你也可以传递自定义的分位数（0 到 1 之间的数值，包含端点）
catt=pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
pd.value_counts(catt)
```




    (-0.0178, 1.29]      400
    (-1.172, -0.0178]    400
    (1.29, 3.305]        100
    (-3.331, -1.172]     100
    dtype: int64



聚合和分组运算时会再次用到cut和qcut，因为这两个离散化函数对分位和分组分析非常重要

#### 检测和过滤异常值


```python
# 过滤或变换异常值（outlier）在很大程度上就是运用数组运算。来看一个含有正态分布数据的DataFrame
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.019061</td>
      <td>-0.014860</td>
      <td>0.043110</td>
      <td>-0.036492</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.013309</td>
      <td>1.039890</td>
      <td>0.996723</td>
      <td>1.016968</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.832478</td>
      <td>-3.173769</td>
      <td>-3.688623</td>
      <td>-2.998333</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.680516</td>
      <td>-0.670257</td>
      <td>-0.646230</td>
      <td>-0.730915</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.061652</td>
      <td>-0.042735</td>
      <td>0.030344</td>
      <td>-0.051680</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.671979</td>
      <td>0.647550</td>
      <td>0.732885</td>
      <td>0.676044</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.034486</td>
      <td>3.567536</td>
      <td>2.897191</td>
      <td>3.170905</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 想要找出某列中绝对值大小超过 3 的值
col=data[2]
col[np.abs(col)>3]
```




    433   -3.528106
    884   -3.688623
    Name: 2, dtype: float64




```python
# 要选出全部含有“超过 3 或 -3 的值”的行，你可以在布尔型DataFrame中使用any方法
data[(np.abs(data)>3).any(1)]
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>1.905776</td>
      <td>-3.173769</td>
      <td>1.262364</td>
      <td>0.363025</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1.733489</td>
      <td>3.009331</td>
      <td>-0.069535</td>
      <td>1.451113</td>
    </tr>
    <tr>
      <th>142</th>
      <td>-3.832478</td>
      <td>1.871240</td>
      <td>-0.697422</td>
      <td>1.042505</td>
    </tr>
    <tr>
      <th>151</th>
      <td>0.719788</td>
      <td>3.567536</td>
      <td>-0.363558</td>
      <td>0.828426</td>
    </tr>
    <tr>
      <th>182</th>
      <td>0.217667</td>
      <td>-3.058076</td>
      <td>-0.659660</td>
      <td>-0.382940</td>
    </tr>
    <tr>
      <th>194</th>
      <td>0.063917</td>
      <td>3.262620</td>
      <td>-0.811974</td>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>393</th>
      <td>3.034486</td>
      <td>-0.260877</td>
      <td>-0.267270</td>
      <td>0.577327</td>
    </tr>
    <tr>
      <th>433</th>
      <td>-0.283554</td>
      <td>1.531852</td>
      <td>-3.528106</td>
      <td>-1.241772</td>
    </tr>
    <tr>
      <th>490</th>
      <td>-3.294577</td>
      <td>0.612080</td>
      <td>0.789589</td>
      <td>0.572930</td>
    </tr>
    <tr>
      <th>678</th>
      <td>-3.229045</td>
      <td>-0.805220</td>
      <td>-0.883356</td>
      <td>-1.503033</td>
    </tr>
    <tr>
      <th>828</th>
      <td>-3.565836</td>
      <td>0.778624</td>
      <td>1.294191</td>
      <td>-0.622613</td>
    </tr>
    <tr>
      <th>859</th>
      <td>1.333673</td>
      <td>-1.166965</td>
      <td>0.059205</td>
      <td>3.170905</td>
    </tr>
    <tr>
      <th>884</th>
      <td>1.912490</td>
      <td>-1.037387</td>
      <td>-3.688623</td>
      <td>0.258966</td>
    </tr>
    <tr>
      <th>996</th>
      <td>0.343030</td>
      <td>-1.531375</td>
      <td>1.289407</td>
      <td>3.047910</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 根据数据的值是正还是负，np.sign(data)可以生成 1 和 -1：
# 根据这些条件，就可以对值进行设置。下面的代码可以将值限制在区间 -3 到 3 以内
data[np.abs(data)>3]=np.sign(data)*3
data.describe()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.017173</td>
      <td>-0.015468</td>
      <td>0.044327</td>
      <td>-0.036711</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.006955</td>
      <td>1.036577</td>
      <td>0.992617</td>
      <td>1.016299</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-3.000000</td>
      <td>-2.998333</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.680516</td>
      <td>-0.670257</td>
      <td>-0.646230</td>
      <td>-0.730915</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.061652</td>
      <td>-0.042735</td>
      <td>0.030344</td>
      <td>-0.051680</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.671979</td>
      <td>0.647550</td>
      <td>0.732885</td>
      <td>0.676044</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.897191</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### 排列和随机采样


```python
# 利用numpy.random.permutation函数可以轻松实现对Series或DataFrame的列的排列工作（permuting，随机重排序）。
≈
df = pd.DataFrame(np.arange(5*4).reshape((5, 4)))
sampler = np.random.permutation(5)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
sampler = np.random.permutation(5)
sampler
#通过需要排列的轴的长度调用permutation，可产生一个表示新顺序的整数数组
```




    array([1, 4, 0, 3, 2])




```python
df.take(sampler)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果不想用替换的方式选取随机子集，可以在Series和DataFrame上使用sample方法
df.sample(n=3)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 要通过替换的方式产生样本（允许重复选择），可以传递replace=True到sample
choices = pd.Series([5, 7, -1, 6, 4])
draws = choices.sample(n=10, replace=True)
draws
```




    3    6
    4    4
    1    7
    0    5
    4    4
    1    7
    3    6
    1    7
    4    4
    2   -1
    dtype: int64



#### 计算指标/哑变量


```python
# 另一种常用于统计建模或机器学习的转换方式是：将分类变量（categorical variable）转换为“哑变量”或“指标矩阵”。

# 如果DataFrame的某一列中含有k个不同的值，则可以派生出一个k列矩阵或DataFrame（其值全为 1 和 0）。
# pandas 有一个get_dummies函数可以实现该功能（其实自己动手做一个也不难）。使用之前的一个DataFrame例子
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                    'data1': range(6)})
pd.get_dummies(df['key'])
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 你可能想给指标DataFrame的列加上一个前缀，以便能够跟其他数据进行合并。get_dummies的prefix参数可以实现该功能
dummies = pd.get_dummies(df['key'], prefix='key')
dummies
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
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy
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
      <th>data1</th>
      <th>key_a</th>
      <th>key_b</th>
      <th>key_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##### 经典案例


```python
# 如果DataFrame中的某行同属于多个分类，则事情就会有点复杂
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('datasets/movielens/movies.dat', sep='::',
                    header=None, names=mnames)
movies[:10]
```

    /Users/snail/opt/anaconda3/lib/python3.8/site-packages/pandas/io/parsers.py:765: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.
      return read_csv(**locals())





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
      <th>movie_id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Heat (1995)</td>
      <td>Action|Crime|Thriller</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tom and Huck (1995)</td>
      <td>Adventure|Children's</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Sudden Death (1995)</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>GoldenEye (1995)</td>
      <td>Action|Adventure|Thriller</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 要为每个genre添加指标变量就需要做一些数据规整操作。首先，我们从数据集中抽取出不同的genre值
all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
genres
```




    array(['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy',
           'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror',
           'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir',
           'Western'], dtype=object)




```python
# 构建指标DataFrame的方法之一是从一个全零DataFrame开始
zero_matrix = np.zeros((len(movies), len(genres)))
dummies = pd.DataFrame(zero_matrix, columns=genres)

```


```python
# 现在，迭代每一部电影，并将dummies各行的条目设为 1。要这么做，我们使用dummies.columns来计算每个类型的列索引
gen = movies.genres[0]
gen.split('|')
dummies.columns.get_indexer(gen.split('|'))
```




    array([0, 1, 2])




```python
# 然后，根据索引，使用.iloc设定值
for i, gen in enumerate(movies.genres):
     indices = dummies.columns.get_indexer(gen.split('|'))
     dummies.iloc[i, indices] = 1
```


```python
# 然后，和以前一样，再将其与movies合并起来
movies_windic = movies.join(dummies.add_prefix('Genre_'))
movies_windic.iloc[0]
```




    movie_id                                       1
    title                           Toy Story (1995)
    genres               Animation|Children's|Comedy
    Genre_Animation                                1
    Genre_Children's                               1
    Genre_Comedy                                   1
    Genre_Adventure                                0
    Genre_Fantasy                                  0
    Genre_Romance                                  0
    Genre_Drama                                    0
    Genre_Action                                   0
    Genre_Crime                                    0
    Genre_Thriller                                 0
    Genre_Horror                                   0
    Genre_Sci-Fi                                   0
    Genre_Documentary                              0
    Genre_War                                      0
    Genre_Musical                                  0
    Genre_Mystery                                  0
    Genre_Film-Noir                                0
    Genre_Western                                  0
    Name: 0, dtype: object



笔记：对于很大的数据，用这种方式构建多成员指标变量就会变得很慢。最好使用更低级的函数，将其写入 NumPy 数组，然后结果包装在DataFrame中


```python
# 一个对统计应用有用的秘诀是：结合get_dummies和诸如cut之类的离散化函数
np.random.seed(12345)
values = np.random.rand(10)
values
```




    array([0.92961609, 0.31637555, 0.18391881, 0.20456028, 0.56772503,
           0.5955447 , 0.96451452, 0.6531771 , 0.74890664, 0.65356987])




```python
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
pd.get_dummies(pd.cut(values, bins))
# 我们用numpy.random.seed，使这个例子具有确定性。本书后面会介绍pandas.get_dummies
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
      <th>(0.0, 0.2]</th>
      <th>(0.2, 0.4]</th>
      <th>(0.4, 0.6]</th>
      <th>(0.6, 0.8]</th>
      <th>(0.8, 1.0]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 字符串操作

#### 字符串对象方法


```python
val = 'a,b,  guido'
val.split(',')
```




    ['a', 'b', '  guido']




```python
# split常常与strip一起使用，以去除空白符（包括换行符）
pieces=[x.strip() for x in val.split(',')]
pieces
```




    ['a', 'b', 'guido']




```python
# 利用加法，可以将这些子字符串以双冒号分隔符的形式连接起来
first, second, third = pieces
first + '::' + second + '::' + third
```




    'a::b::guido'




```python
# 但这种方式并不是很实用。一种更快更符合 Python 风格的方式是，向字符串"::"的join方法传入一个列表或元组
'::'.join(pieces)
```




    'a::b::guido'




```python
val.find(':') # 未找到，返回-1
```




    -1




```python
val.index('a')
# 注意find和index的区别：如果找不到字符串，index将会引发一个异常（而不是返回 -1）
val.index(':')
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-222-b5c24c943f7f> in <module>
          1 val.index('a')
          2 # 注意find和index的区别：如果找不到字符串，index将会引发一个异常（而不是返回 -1）
    ----> 3 val.index(':')
    

    ValueError: substring not found



```python
# 与此相关，count可以返回指定子串的出现次数
val.count(',')
```




    2




```python
# replace用于将指定模式替换为另一个模式。通过传入空字符串，它也常常用于删除模式
val.replace(',', '::')
val.replace(',', '')
```




    'a::b::  guido'



#### 正则表达式



```python
# 正则表达式，常称作 regex 。。Python 内置的re模块负责对字符串应用正则表达式
# re模块的函数可以分为三个大类：模式匹配、替换以及拆分。当然，它们之间是相辅相成的。
# 一个正则表达式描述了需要在文本中定位的一个模式，它可以用于许多目的。
#假设我想要拆分一个字符串，分隔符为数量不定的一组空白符（制表符、空格、换行符等）。
# 描述一个或多个空白符的正则表达式是\s+
import re
text= "foo    bar\t baz  \tqux"
re.split('\s+',text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
# 调用re.split('\s+',text)时，正则表达式会先被编译，然后再在text上调用其split方法。
# 你可以用re.compile自己编译正则表达式以得到一个可重用的正则表达式对象
regex=re.compile('\s+')
regex.split(text)
```




    ['foo', 'bar', 'baz', 'qux']




```python
# 如果只希望得到匹配regex的所有模式，则可以使用findall方法
regex.findall(text)
# 如果想避免正则表达式中不需要的转义（\），则可以使用原始字符串字面量如r'C:\x'（也可以编写其等价式'C:\\x'
```




    ['    ', '\t ', '  \t']




```python
# 如果打算对许多字符串应用同一条正则表达式，
# 强烈建议通过re.compile创建regex对象。这样将可以节省大量的 CPU 时间
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
```




    ['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', 'ryan@yahoo.com']




```python
m=regex.search(text)
m
#search返回的是文本中第一个电子邮件地址（以特殊的匹配项对象形式返回）。
#对于上面那个regex，匹配项对象只能告诉我们模式在原字符串中的起始和结束位置
#text[m.start():m.end()]
```




    <re.Match object; span=(5, 20), match='dave@google.com'>




```python
# regex.match则将返回None，因为它只匹配出现在字符串开头的模式
print(regex.match(text))
```

    None



```python
# 相关的，sub方法可以将匹配到的模式替换为指定字符串，并返回所得到的新字符串
print(regex.sub('REDACT', text))
```

    Dave REDACT
    Steve REDACT
    Rob REDACT
    Ryan REDACT
    



```python
# 假设你不仅想要找出电子邮件地址，还想将各个地址分成 3 个部分：
# 用户名、域名以及域后缀。要实现此功能，只需将待分段的模式的各部分用圆括号包起来即可
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match('wesm@bright.net')
m.group(0) # m.groups()
```




    'wesm@bright.net'




```python
# 对于带有分组功能的模式，findall会返回一个元组列表
regex.findall(text)
```




    [('dave', 'google', 'com'),
     ('steve', 'gmail', 'com'),
     ('rob', 'gmail', 'com'),
     ('ryan', 'yahoo', 'com')]




```python
# sub还能通过诸如\1、\2之类的特殊符号访问各匹配项中的分组。符号\1对应第一个匹配的组，\2对应第二个匹配的组，以此类推：
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))
```

    Dave Username: dave, Domain: google, Suffix: com
    Steve Username: steve, Domain: gmail, Suffix: com
    Rob Username: rob, Domain: gmail, Suffix: com
    Ryan Username: ryan, Domain: yahoo, Suffix: com
    


#### pandas 的向量化字符串函数


```python
# 清理待分析的散乱数据时，常常需要做一些字符串规整化工作。更为复杂的情况是，含有字符串的列有时还含有缺失数据
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
            'Rob': 'rob@gmail.com', 'Wes': np.nan}
data =pd.Series(data)
data.isnull()
```




    Dave     False
    Steve    False
    Rob      False
    Wes       True
    dtype: bool




```python
# 通过data.map，所有字符串和正则表达式方法都能被应用于（传入 lambda 表达式或其他函数）各个值，但是如果存在 NA（null）就会报错。
# 为了解决这个问题，Series有一些能够跳过 NA 值的面向数组方法，进行字符串操作。
# 通过Series的str属性即可访问这些方法。
data.str.contains('gmail')
```




    Dave     False
    Steve     True
    Rob       True
    Wes        NaN
    dtype: object




```python
pattern
```




    '([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\\.([A-Z]{2,4})'




```python
data.str.findall(pattern, flags=re.IGNORECASE)
```




    Dave     [(dave, google, com)]
    Steve    [(steve, gmail, com)]
    Rob        [(rob, gmail, com)]
    Wes                        NaN
    dtype: object




```python
# 有两个办法可以实现向量化的元素获取操作：要么使用str.get，要么在str属性上使用索引
matches = data.str.match(pattern, flags=re.IGNORECASE)
matches
```




    Dave     True
    Steve    True
    Rob      True
    Wes       NaN
    dtype: object




```python
# 要访问嵌入列表中的元素，我们可以传递索引到这两个函数中
data.str.get(1)
```




    Dave       a
    Steve      t
    Rob        o
    Wes      NaN
    dtype: object




```python
data.str[0]
```




    Dave       d
    Steve      s
    Rob        r
    Wes      NaN
    dtype: object




```python
# 可以利用这种方法对字符串进行截取
data.str[:5]
```




    Dave     dave@
    Steve    steve
    Rob      rob@g
    Wes        NaN
    dtype: object




```python

```
