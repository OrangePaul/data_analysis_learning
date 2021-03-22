### 数据加载、存储与文件格式
输入输出通常可以划分为几个大类：读取文本文件和其他更高效的磁盘存储格式，加载数据库中的数据，利用 Web API 操作网络资源。

### 6.1 读写文本格式的数据
pandas 提供了一些用于将表格型数据读取为DataFrame对象的函数。其中read_csv和read_table可能会是今后用得最多的

read_csv

read_table

read_excel

read_json

read_sql

#### 索引：将一个或多个列当做返回的DataFrame处理，以及是否从文件、用户获取列名。
#### 类型推断和数据转换：包括用户定义值的转换、和自定义的缺失值标记列表等。
#### 日期解析：包括组合功能，比如将分散在多个列中的日期时间信息组合成结果中的单个列。
#### 迭代：支持对大文件进行逐块迭代。
#### 不规整数据问题：跳过一些行、页脚、注释或其他一些不重要的东西（比如由成千上万个逗号隔开的数值数据）


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
!cat examples/ex1.csv
```

    a,b,c,d,message
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo


```python
# df=pd.read_csv('examples/ex1.csv')
df=pd.read_table('examples/ex1.csv',sep=',')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
!cat examples/ex2.csv
```

    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo

#### 无标题行文件


```python
# 没有标题行的文件，可以pandas为其默认分配列名，可以自己设定
# df2=pd.read_csv('examples/ex2.csv',header=None)\
# df2=pd.read_csv('examples/ex2.csv',header=2) 可以设置header 选择数据的里的第N行作为标题
df2=pd.read_csv('examples/ex2.csv',names=['a', 'b', 'c', 'd', 'message'])
df2
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 假设你希望将message列做成DataFrame的索引。你可以明确表示要将该列放到索引 4 的位置上，也可以通过index_col参数指定"message"：
names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('examples/ex2.csv', names=names, index_col='message')
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
    <tr>
      <th>message</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hello</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>world</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
#如果希望将多个列做成一个层次化索引，只需传入由列编号或列名组成的列表即可
!cat examples/csv_mindex.csv
```

    key1,key2,value1,value2
    one,a,1,2
    one,b,3,4
    one,c,5,6
    one,d,7,8
    two,a,9,10
    two,b,11,12
    two,c,13,14
    two,d,15,16



```python
pd.read_csv('examples/csv_mindex.csv',index_col=['key1','key2'])
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
      <th></th>
      <th>value1</th>
      <th>value2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">one</th>
      <th>a</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>b</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>c</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>d</th>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">two</th>
      <th>a</th>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>b</th>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>c</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>d</th>
      <td>15</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
list(open('examples/ex3.txt'))
# 有些表格可能不是用固定的分隔符去分隔字段的（比如空白符或其它模式）
```




    ['            A         B         C\n',
     'aaa -0.264438 -1.026059 -0.619500\n',
     'bbb  0.927272  0.302904 -0.032399\n',
     'ccc -0.264273 -0.386314 -0.217601\n',
     'ddd -0.871858 -0.348382  1.100491\n']



#### 不同的空白字符间隔


```python
# 虽然可以手动对数据进行规整，这里的字段是被数量不同的空白字符间隔开的。
# 这种情况下，你可以传递一个正则表达式作为read_table的分隔符。可以用正则表达式表达为\s+
result = pd.read_table('examples/ex3.txt', sep='\s+')
result
# 这里，由于列名比数据行的数量少，所以read_table推断第一列应该是DataFrame的索引
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aaa</th>
      <td>-0.264438</td>
      <td>-1.026059</td>
      <td>-0.619500</td>
    </tr>
    <tr>
      <th>bbb</th>
      <td>0.927272</td>
      <td>0.302904</td>
      <td>-0.032399</td>
    </tr>
    <tr>
      <th>ccc</th>
      <td>-0.264273</td>
      <td>-0.386314</td>
      <td>-0.217601</td>
    </tr>
    <tr>
      <th>ddd</th>
      <td>-0.871858</td>
      <td>-0.348382</td>
      <td>1.100491</td>
    </tr>
  </tbody>
</table>
</div>



#### 存在无用行文本


```python
!cat examples/ex4.csv
```

    # hey!
    a,b,c,d,message
    # just wanted to make things more difficult for you
    # who reads CSV files with computers, anyway?
    1,2,3,4,hello
    5,6,7,8,world
    9,10,11,12,foo


```python
pd.read_csv('examples/ex4.csv', skiprows=[0,2,3])
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
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>



#### 缺失值存在的文件


```python
# 缺失值处理是文件解析任务中的一个重要组成部分。
# 缺失数据经常是要么没有（空字符串），要么用某个标记值表示。
# 默认情况下，pandas 会用一组经常出现的标记值进行识别，比如 NA 及NULL
!cat examples/ex5.csv
```

    something,a,b,c,d,message
    one,1,2,3,4,NA
    two,5,6,,8,world
    three,9,10,11,12,foo


```python
result = pd.read_csv('examples/ex5.csv')
result
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.isnull()
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# na_values可以用一个列表或集合的字符串表示缺失值
result = pd.read_csv('examples/ex5.csv', na_values=['NULL'])
result
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 字典的各列可以使用不同的 NA 标记值
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
pd.read_csv('examples/ex5.csv',na_values=sentinels)
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 逐块读取文本文件
关键是chunksize分块和 Series.add 用到fill_value=0


```python
# 在处理很大的文件时，或找出大文件中的参数集以便于后续处理时，你可能只想读取文件的一小部分或逐块对文件进行迭代。
# 在看大文件之前，我们先设置 pandas 显示地更紧些：
pd.options.display.max_rows = 10
result = pd.read_csv('examples/ex6.csv',nrows=5)# 通过nrows指定读的行数
result
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
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.467976</td>
      <td>-0.038649</td>
      <td>-0.295344</td>
      <td>-1.824726</td>
      <td>L</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.358893</td>
      <td>1.404453</td>
      <td>0.704965</td>
      <td>-0.200638</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.501840</td>
      <td>0.659254</td>
      <td>-0.421691</td>
      <td>-0.057688</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.204886</td>
      <td>1.074134</td>
      <td>1.388361</td>
      <td>-0.982404</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.354628</td>
      <td>-0.133116</td>
      <td>0.283763</td>
      <td>-0.837063</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 逐块读取文件，可以指定chunksize（行数）
chunker = pd.read_csv('examples/ex6.csv', chunksize=1000)
chunker
```




    <pandas.io.parsers.TextFileReader at 0x1174d08b0>




```python
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0) #计算看 in[64]

tot = tot.sort_values(ascending=False)
tot[:10]
```

    <ipython-input-64-d5548d1559f2>:1: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
      tot = pd.Series([])





    E    368.0
    X    364.0
    L    346.0
    O    343.0
    Q    340.0
    M    338.0
    J    337.0
    F    335.0
    K    334.0
    H    330.0
    dtype: float64




```python
obj22 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj23 = pd.Series([8, 4, -1, 2], index=['d', 'b', 'e', 'f'])
obj22.add(obj23,fill_value=0) # 缺失值NaN与任何值相加的结果均为NaN，所以这就是为什么要用到fill_value的原因啦
```




    a    -5.0
    b    11.0
    c     3.0
    d    12.0
    e    -1.0
    f     2.0
    dtype: float64



#### 将数据写出到文本格式


```python
# 数据也可以被输出为分隔符格式的文本。
data = pd.read_csv('examples/ex5.csv')
import sys
data
# data.to_csv('examples/out.csv') # 利用DataFrame的to_csv方法，我们可以将数据写到一个以逗号分隔的文件
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
      <th>something</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>1</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>two</td>
      <td>5</td>
      <td>6</td>
      <td>NaN</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>three</td>
      <td>9</td>
      <td>10</td>
      <td>11.0</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 缺失值在输出结果中会被表示为空字符串。你可能希望将其表示为别的标记值
data.to_csv(sys.stdout, na_rep='NULL')
```

    ,something,a,b,c,d,message
    0,one,1,2,3.0,4,NULL
    1,two,5,6,NULL,8,world
    2,three,9,10,11.0,12,foo



```python
data.to_csv(sys.stdout, sep='|')
```

    |something|a|b|c|d|message
    0|one|1|2|3.0|4|
    1|two|5|6||8|world
    2|three|9|10|11.0|12|foo



```python
# 如果没有设置其他选项，则会写出行和列的标签。当然，它们也都可以被禁用
data.to_csv(sys.stdout, index=False, header=False)
```

    one,1,2,3.0,4,
    two,5,6,,8,world
    three,9,10,11.0,12,foo



```python
# 你还可以只写出一部分的列，并以你指定的顺序排列
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])
```

    a,b,c
    1,2,3.0
    5,6,
    9,10,11.0



```python
# Series也有一个to_csv方法
In [50]: dates = pd.date_range('1/1/2000', periods=7)

In [51]: ts = pd.Series(np.arange(7), index=dates)

In [52]: ts.to_csv('examples/tseries.csv')

In [53]: !cat examples/tseries.csv
2000-01-01,0
2000-01-02,1
2000-01-03,2
2000-01-04,3
2000-01-05,4
2000-01-06,5
2000-01-07,6
```

### 处理分隔符格式
大部分存储在磁盘上的表格型数据都能用pandas.read_table进行加载。然而，有时还是需要做一些手工处理。由于接收到含有畸形行的文件而使read_table出毛病的情况并不少见


```python
!cat examples/ex7.csv
```

    "a","b","c"
    "1","2","3"
    "1","2","3"



```python
pd.read_csv('examples/ex7.csv')
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
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
import csv
with open('examples/ex7.csv') as f:
    lines=list(csv.reader(f))
header,values=lines[0],lines[1:] 
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict
```




    {'a': ('1', '1'), 'b': ('2', '2'), 'c': ('3', '3')}



### JSON数据
除其空值null和一些其他的细微差别（如列表末尾不允许存在多余的逗号）之外，JSON 非常接近于有效的 Python 代码。基本类型有对象（字典）、数组（列表）、字符串、数值、布尔值以及null。对象中所有的键都必须是字符串。许多 Python 库都可以读写 JSON 数据


```python
obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
```


```python
# 通过json.loads即可将 JSON 字符串转换成 Python 形式
import json
result=json.loads(obj)
# json.dumps则将 Python 对象转换成 JSON 格式
asjson=json.dumps(result)
```


```python
result
```




    {'name': 'Wes',
     'places_lived': ['United States', 'Spain', 'Germany'],
     'pet': None,
     'siblings': [{'name': 'Scott', 'age': 30, 'pets': ['Zeus', 'Zuko']},
      {'name': 'Katie', 'age': 38, 'pets': ['Sixes', 'Stache', 'Cisco']}]}




```python
# 将（一个或一组）JSON 对象转换为DataFrame
# 最简单方便的方式是：向DataFrame构造器传入一个字典的列表（就是原先的 JSON 对象），并选取数据字段的子集
siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
siblings
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Scott</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Katie</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>




```python
result['siblings']
```




    [{'name': 'Scott', 'age': 30, 'pets': ['Zeus', 'Zuko']},
     {'name': 'Katie', 'age': 38, 'pets': ['Sixes', 'Stache', 'Cisco']}]




```python
# !cat examples/example.json
!cat examples/example.json
```

    [{"a": 1, "b": 2, "c": 3},
     {"a": 4, "b": 5, "c": 6},
     {"a": 7, "b": 8, "c": 9}]



```python
# pandas.read_json可以自动将特别格式的 JSON 数据集转换为Series或DataFrame
pd.read_json('examples/example.json')
# 将数据从 pandas 输出到 JSON，可以使用to_json方法
data.to_json()
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
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>8</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



### 读取Excel文件
pandas 的ExcelFile类或pandas.read_excel函数支持读取存储在 Excel 2003（或更高版本）中的表格型数据。这两个工具分别使用扩展包 xlrd 和 OpenPyXL 读取 XLS 和 XLSX 文件。


```python
frame=pd.read_excel('examples/ex1.xlsx','Sheet1')
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
      <th>Unnamed: 0</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 要使用ExcelFile，通过传递 xls 或 xlsx 路径创建一个实例
xlsx = pd.ExcelFile('examples/ex1.xlsx')
# 存储在表单中的数据可以read_excel读取到DataFrame
#（原书这里写的是用parse解析，但代码中用的是read_excel，是个笔误：只换了代码，没有改文字
pd.read_excel(xlsx, 'Sheet1')
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
      <th>Unnamed: 0</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>hello</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>world</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 如果要将 pandas 数据写入为 Excel 格式，你必须首先创建一个ExcelWriter，然后使用 pandas 对象的to_excel方法将数据写入到其中
writer = pd.ExcelWriter('examples/ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()
```


```python
# 你还可以不使用ExcelWriter，而是传递文件的路径到to_excel
frame.to_excel('examples/ex2.xlsx')
```
