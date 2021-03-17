## Python 的数据结构、函数和文件

### 元组
元组是一个固定长度，不可改变的 Python 序列对象。创建元组的简单方式，是用逗号分隔一列值


```python
tup =4,5,6
print(type(tup))
nested_tuple=(4,5,6),(7,8)
nested_tuple
```

    <class 'tuple'>





    ((4, 5, 6), (7, 8))



#### 用tuple可以将任意序列或迭代器转换成元组
如果元组中的某个对象是可变的，比如列表，可以在原位进行修改：


```python
tup = tuple(['foo', [1, 2], True])
tup[1].append(3)
tup
```




    ('foo', [1, 2, 3], True)



可以用加号运算符将元组串联起来


```python
(4, None, 'foo') + (6, 0) + ('bar',)
```




    (4, None, 'foo', 6, 0, 'bar')



元组乘以一个整数，像列表一样，会将几个元组的复制串联起来：


```python
('foo', 'bar') * 4
```




    ('foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar')



### 元组拆包 
#### 你想将元组赋值给类似元组的变量，Python 会试图拆分等号右边的值
#### 同理嵌套元组也可以拆包


```python
tup =(4,5,(6,7))
a,b,c=tup
c
```




    (6, 7)




```python
#使用这个功能，你可以很容易地替换变量的名字
a,b=1,2
a,b=b,a
print(a,b)
```

    2 1



```python
#变量拆分常用来迭代元组或列表序列：
seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
for a,b,c in seq:
    print(a)
    print(b)
    print(c)
    print('a={0},b={1},c={2}'.format(a,b,c))
```

    1
    2
    3
    a=1,b=2,c=3
    4
    5
    6
    a=4,b=5,c=6
    7
    8
    9
    a=7,b=8,c=9



```python
#拆包工具*rest
values=[1,2,3,4,5]
a,b,*rest=values
print(rest)
#rest的部分是想要舍弃的部分，rest的名字不重要。
#作为惯用写法，许多 Python 程序员会将不需要的变量使用下划线
a, b, *_ = values
```




    [3, 4, 5]



### 添加和删除元素
可以用append在列表末尾添加元素
insert可以在特定的位置插入元素
##### insert耗费的计算量大，因为对后续元素的引用必须在内部迁移，以便为新元素提供空间。如果要在序列的头部和尾部插入元素，你可能需要使用collections.deque，一个双尾部队列。
insert的逆运算是pop，它移除并返回指定位置的元素
可以用remove去除某个值，remove会先寻找第一个值并除去
用in可以检查列表是否包含某个值

### 串联和组合列表


```python
# 通过加法将列表串联的计算量较大，因为要新建一个列表，并且要复制对象。
# 用extend追加元素，尤其是到一个大列表中，更为可取
x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])
x
```




    [4, None, 'foo', 7, 8, (2, 3)]



### 二分搜索和维护已排序的列表
bisect模块支持二分查找，和向已排序的列表插入值。bisect.bisect可以找到插入值后仍保证排序的位置，bisect.insort是向这个位置插入值


```python
import bisect
c= [1,2,2,2,3,4,7]
bisect.bisect(c,2) #4
bisect.bisect(c,4) #6
bisect.insort(c,6) #找到对应的索引插入
c
```




    [1, 2, 2, 2, 3, 4, 6, 7]



### 切片
用切边可以选取大多数序列类型的一部分，切片的基本形式是在方括号中使用start:stop
切片也可以被序列赋值
#### seq[3:4] = [6, 3]
切片的起始元素是包括的，不包含结束元素。因此，结果中包含的元素个数是stop - start
在第二个冒号后面使用step，可以隔n个取一个元素


```python
# 一个聪明的方法是使用-1，它可以将列表或元组颠倒过来
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[::-1]
```




    [1, 0, 6, 5, 7, 3, 2, 7]



### 序列函数
迭代一个序列时，你可能想跟踪当前项的序号
Python 内建了一个enumerate函数，可以返回(i, value)元组序列


```python
collection=[1,44,7]
for i, value in enumerate(collection):
    print(i,value)
```

    0 1
    1 44
    2 7



```python
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i,v in enumerate(some_list):
    mapping[i]=v
mapping
```




    {0: 'foo', 1: 'bar', 2: 'baz'}



#### sorted函数
可以从任意序列的元素返回一个新的排好序的列表



```python
sorted([7, 1, 2, 6, 0, 3, 2])
sorted('horse race')
```




    [' ', 'a', 'c', 'e', 'e', 'h', 'o', 'r', 'r', 's']




```python
# zip可以将多个列表、元组或其它序列成对组合成一个元组列表
# zip可以处理任意多的序列，元素的个数取决于最短的序列
seq1 = ['foo', 'bar', 'baz']
seq2 = ['one', 'two', 'three']
seq3 =['true','false']
zipped = zip(seq1, seq2,seq3)
list(zipped)
```




    [('foo', 'one', 'true'), ('bar', 'two', 'false')]




```python
# zip也可以用来解压序列，把行的列表转化为列的列表
pitchers = [('Nolan', 'Ryan'), 
            ('Roger', 'Clemens'),
            ('Schilling', 'Curt')]
first,last=zip(*pitchers)
first
```




    ('Nolan', 'Roger', 'Schilling')




```python
# reversed可以从后向前迭代一个序列
list(reversed(range(10)))
```




    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]



### 字典
字典可能是 Python 最为重要的数据结构。它更为常见的名字是哈希映射或关联数组。它是键值对的大小可变集合，键和值都是 Python 对象。创建字典的方法之一是使用尖括号，用冒号分隔键和值



```python
# 用检查列表和元组是否包含某个值的方法，检查字典中是否包含某个键：
d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}
'b' in d1 #True
#可以用del关键字或pop方法（返回值的同时删除键）删除值
del d1['b']
ret = d1.pop('a')
ret
```




    'some value'



### keys和values是字典的键和值的迭代器方法。虽然键值对没有顺序，这两个方法可以用相同的顺序输出键和值


```python
#用update方法可以将一个字典与另一个融合
# update方法是原地改变字典，因此任何传递给update的键的旧的值都会被舍弃。
d1={'a':'some','b':'fasdfafa'}
d1.update({'b' : 'foo', 'c' : 12})
d1
```




    {'a': 'some', 'b': 'foo', 'c': 12}




```python
# 创建字典方法一：
# mapping = {}
# for key, value in zip(key_list, value_list):
#     mapping[key] = value
# 因为字典本质上是 2 元元组的集合，字典可以接受 2 元元组的列表：
# 方法二
mapp = dict(zip(range(5), reversed(range(5))))# 相当于dict(两个同长度元组)
mapp
```




    {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}




```python
#字典的方法get和pop可以取默认值进行返回
words = {'apple': 'bat', 'bar':'atom','book':'1'}
# value=words.pop('bar','None')
value=words.get('bar','None')
value
```




    'atom'




```python
# setdefault用法 和get()相似 如果键不存在字典中，将会添加键并将值设为默认值
from collections import defaultdict
# collections模块有一个很有用的类，defaultdict，
# 它可以进一步简化上面。传递类型或函数以生成每个位置的默认值：
by_letter = defaultdict(list)
words = ['apple', 'bat', 'bar', 'atom', 'book']
# setdefault 如果不存在会在原字典里添加一个 key:default_value 并返回 default_value
# get 找不到 key 的时候不会修改原字典，只返回 default_value。
# 若要修改字典 dic.setdefault(key,default_value) 
# 等同于 dic[key] = dic.get(key,default_value)
for word in words:
    letter = word[0]
    by_letter.setdefault(letter, []).append(word)
by_letter
```




    defaultdict(list, {'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']})



### defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值


```python
# defaultdict接受一个工厂函数作为参数，如下来构造
from collections import defaultdict

dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)
dict1[2] ='two'

print(dict1[1])
print(dict2[1])
print(dict3[1])
print(dict4[1])
```

    0
    set()
    
    []



```python
from collections import defaultdict
words = ['apple', 'bat', 'bar', 'atom', 'book']
by_letter = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
by_letter
```




    defaultdict(list, {'a': ['apple', 'atom'], 'b': ['bat', 'bar', 'book']})



#### 字典的值可以是任意 Python 对象，而键通常是不可变的标量类型（整数、浮点型、字符串）或元组（元组中的对象必须是不可变的）。这被称为“可哈希性”。可以用hash函数检测一个对象是否是可哈希的（可被用作字典的键）


```python
print(hash('string'))
print(hash((1,2,(2,3))))
hash((1,2,[2,3]))
#要用列表当做键，一种方法是将列表转化为元组，只要内部元素可以被哈希，它也就可以被哈希：
```

    2581889065171836302
    -9209053662355515447



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-94-90372972370e> in <module>
          1 print(hash('string'))
          2 print(hash((1,2,(2,3))))
    ----> 3 hash((1,2,[2,3]))
    

    TypeError: unhashable type: 'list'


### 集合
集合是无序的不可重复的元素的集合。你可以把它当做字典，但是只有键没有值。可以用两种方式创建集合：通过set函数或使用尖括号set语句：


```python
set([2, 2, 2, 1, 3, 3])
```




    {1, 2, 3}



 ### 列表、集合和字典的推导式



```python
# 列表推导式
#[expr for val in collection if condition]
# 等同于
# result = []
# for val in collection:
#     if condition:
#         result.append(expr)
str1 = ['a', 'as', 'bat', 'car', 'dove', 'python']
[u.upper() for u in str1 if len(u) > 2]

```




    ['BAT', 'CAR', 'DOVE', 'PYTHON']




```python
# 字典推导式
dict_comp = {key-expr : value-expr for value in collection if condition}

# 集合推导式
set_comp = {expr for value in collection if condition}
```


```python
str1 = ['a', 'as', 'bat', 'car', 'dove', 'python']
unique_lengths = {len(x) for x in str1}
unique_lengths
# map函数可以进一步
set(map(len, strings))
```




    {1, 2, 3, 4, 6}




```python
# 嵌套列表推导式
all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
           ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
names_of_interest = []
for names in all_data:
    enough_es = [name for name in names if name.count('e') >= 2]
    names_of_interest.extend(enough_es)
#等同于
result = [name for names in all_data for name in names
             if name.count('e') >= 2]
result 
```




    ['Steven']



## 函数
### 函数是 Python 中最主要也是最重要的代码组织和复用手段。作为最重要的原则，如果你要重复使用相同或非常类似的代码，就需要写一个函数


```python
# 函数使用def关键字声明，用return关键字返回值
def my_function(x, y, z=1.5):
    if z > 1:
        return z * (x + y)
    else:
        return z / (x + y)
# 同时拥有多条return语句也是可以的。如果到达函数末尾时没有遇到任何一条return语句，则返回None。

# 函数可以有一些位置参数（positional）和一些关键字参数（keyword）。
# 关键字参数通常用于指定默认值或可选参数。
# 在上面的函数中，x和y是位置参数，而z则是关键字参数
```

### 命名空间、作用域，和局部函数
函数可以访问两种不同作用域中的变量：全局（global）和局部（local）。Python 有一种更科学的用于描述变量作用域的名称，即命名空间（namespace）。任何在函数中赋值的变量默认都是被分配到局部命名空间（local namespace）中的。局部命名空间是在函数被调用时创建的，函数参数会立即填入该命名空间。在函数执行完毕之后，局部命名空间就会被销毁


```python
# 返回多个值
def f():
    a = 5
    b = 6
    c = 7
    return a, b, c
    #return {'a' : a, 'b' : b, 'c' : c} 返回字典
#a,b,c=f()
return_value = f() #返回3元远组
print(return_value)
```

    (5, 6, 7)



```python
# map() 会根据提供的函数对指定序列做映射。

# 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
```


```python
states = ['   Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda',
              'south   carolina##', 'West virginia?']
#方法一 处理字符串
import re

def clean_strings(strings):
    result = []
    for value in strings:
        value = value.strip()
        value = re.sub('[!#?]', '', value)
        value = value.title()
        result.append(value)
    return result
clean_strings(states)
```




    ['Alabama',
     'Georgia',
     'Georgia',
     'Georgia',
     'Florida',
     'South   Carolina',
     'West Virginia']




```python
# 方法二 处理字符串 将需要在一组给定字符串上执行的所有运算做成一个列表：
def remove_punctuation(value):
    return re.sub('[!#?]', '', value)

clean_ops = [str.strip, remove_punctuation, str.title]

def clean_strings(strings, ops):
    result = []
    for value in strings:
        for function in ops:
            value = function(value)
        result.append(value)
    return result
clean_strings(states, clean_ops)
```




    ['Alabama',
     'Georgia',
     'Georgia',
     'Georgia',
     'Florida',
     'South   Carolina',
     'West Virginia']




```python
#方法三 处理字符串 运用上文的map函数
clean_ops = [str.strip, remove_punctuation, str.title]
for x in map(remove_punctuation,states):
    print(x)

```

       Alabama 
    Georgia
    Georgia
    georgia
    FlOrIda
    south   carolina
    West virginia


  ### lambda函数
  该语句的结果就是返回值


```python
ints = [4, 0, 1, 5, 6]
# lambda匿名函数的格式：冒号前是参数，可以有多个，用逗号隔开，冒号右边的为表达式。
# 其实lambda返回值是一个函数的地址，也就是函数对象。
p=lambda x:x+1
p(2)
```




    3




```python
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
strings
```




    ['aaaa', 'foo', 'abab', 'bar', 'card']




```python
# 假如a是一个由元组构成的列表，这时候就麻烦了，我们需要用到参数key，也就是关键词，
# 看下面这句命令，lambda是一个隐函数，是固定写法，不要写成别的单词；
# x表示列表中的一个元素，在这里，表示一个元组，
# x只是临时起的一个名字，你可以使用任意的名字；
# x[0]表示元组里的第一个元素，当然第二个元素就是x[1]；
#所以这句命令的意思就是按照列表中第一个元素排序 
#sorted(iterable,key,reverse)
a=[('a',0),('b',4),('c',2)]
sorted(a,key=lambda _:_[0])
```




    [('a', 0), ('b', 4), ('c', 2)]



### 柯里化：部分参数应用
#### 更多是出现在函数式编程中
柯里化（currying）是一个有趣的计算机科学术语，它指的是通过“部分参数应用”（partial argument application）从现有函数派生出新函数的技术


```python
def add_numbers(x, y):
    return x + y
#派生出只有1个参数的函数add_five
add_five=lambda y:add_numbers(5,y)
add_five(4)
# add_numbers的第二个参数称为“柯里化的”（curried）。
# 因为我们其实就只是定义了一个可以调用现有函数的新函数而已。
# 内置的functools模块可以用partial函数将此过程简化
from functools import partial
add_five = partial(add_numbers, 5)

```

### 生成器 generator
能以一种一致的方式对序列进行迭代（比如列表中的对象或文件中的行）是 Python 的一个重要特点。这是通过一种叫做迭代器协议（iterator protocol，它是一种使对象可迭代的通用方式）的方式实现的，一个原生的使对象可迭代的方法。比如说，对字典进行迭代可以得到其所有的键：




```python
some_dict = {'a': 1, 'b': 2, 'c': 3}
for key in some_dict:    
    print(key)
```

    a
    b
    c


#### 当你编写for key in some_dict时，Python 解释器首先会尝试从some_dict创建一个迭代器：
#### 迭代器是一种特殊对象，它可以在诸如for循环之类的上下文中向 Python 解释器输送对象
#### 大部分能接受列表之类的对象的方法也都可以接受任何可迭代对象。比如min、max、sum等内置方法以及list、tuple等类型构造器


```python
some_dict = {'a': 1, 'b': 2, 'c': 3}
# 这是底层 写for key in some_dict时，
# Python 解释器首先会尝试从some_dict创建一个迭代器
dict_iterator = iter(some_dict)
dict_iterator
```




    <dict_keyiterator at 0x10e555b30>




```python
list(dict_iterator)
```




    ['a', 'b', 'c']




```python
# 生成器（generator）是构造新的可迭代对象的一种简单方式。
# 一般的函数执行之后只会返回单个值，而生成器则是以延迟的方式返回一个值序列，
# 即每返回一个值之后暂停，直到下一个值被请求时再继续。
# 要创建一个生成器，只需将函数中的return替换为yeild即可
def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))
    for i in range(1, n + 1):
        yield i ** 2
```


```python
# 调用该生成器时，没有任何代码会被立即执行：
gen=squares() #squares不会真的执行，而是得到一个生成器gen
gen
```




    <generator object squares at 0x10ebcb5f0>




```python
for x in gen:
    print(x,end=' ')
```

    Generating squares from 1 to 100
    1 4 9 16 25 36 49 64 81 100 

### 另一种更简洁的构造生成器的方法是使用生成器表达式（generator expression）。
#### 这是一种类似于列表、字典、集合推导式的生成器。其创建方式为，把列表推导式两端的方括号改成圆括号


```python
# def _make_gen():
#     for x in range(100):
#         yield x ** 2
# gen = _make_gen()
# 上下等价
gen = (x ** 2 for x in range(100))
sum(gen)
```




    328350



#### 容器是可迭代对象，可迭代对象调用iter()，可以得到一个迭代器。
#### 迭代器可以通过next()函数来得到下一个元素，从而支持遍历。
#### 生成器是一个特殊的迭代器（反之不成立）合理使用生成器可以减低内存占用，优化程序结构


```python
# 1, 生成器的样子就是一个普通的函数，只不过return关键词被yield取代了
# 2, 当调用这个“函数”的时候，它会立即返回一个迭代器，而不立即执行函数内容，直到调用其返回迭代器的next方法是才开始执行，直到遇到yield语句暂停。
# 3, 继续调用生成器返回的迭代器的next方法，恢复函数执行，直到再次遇到yield语句
# 4, 如此反复，一直到遇到StopIteration
def gFun():
    print('before hello')
    yield 'hello'
    print('after hello')

a = gFun() # 调用生成器函数，返回一个迭代器并赋给a
# 简要理解：yield就是 return 返回一个值，并且记住这个返回的位置，
# 下次迭代就从这个位置后开始
print(a) # <generator object gFun at 0x104cd2a40> 得到一个生成器对象(迭代器)
print(a.__next__())
# before hello
# hello
print(a.__next__())
# after hello
# StopIteration
```

    <generator object gFun at 0x10ed5e740>
    before hello
    hello
    after hello



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-205-2b71b864f46f> in <module>
         14 # before hello
         15 # hello
    ---> 16 print(a.__next__())
         17 # after hello
         18 # StopIteration


    StopIteration: 


### itertools模块
标准库itertools模块中有一组用于许多常见数据算法的生成器。例如，groupby可以接受任何序列和一个函数。它根据函数的返回值对序列中的连续元素进行分组
### 如果iterable在多次连续迭代中生成了同一项，则会定义一个组，如果将此函数应用一个分类列表，那么分组将定义该列表中的所有唯一项，key（如果已提供）是一个函数，应用于每一项，如果此函数存在返回值，该值将用于后续项而不是该项本身进行比较，此函数返回的迭代器生成元素(key, group)，其中key是分组的键值，group是迭代器，生成组成该组的所有项。


```python
from itertools import groupby
# groupby()的作用就是把可迭代对象中相邻的重复元素挑出来放一起
test=[(1,5),(1,4),(1,3),(1,2),(2,1),(2,3),(2,4),(3,5)]
temp= groupby(test,key=lambda x:x[0])
# for a,b in temp:
#     print(a,list(b))# 为什么这里要用list()函数呢？
print(temp)
```

    <itertools.groupby object at 0x10eede810>


### 错误和异常处理
### 假如想优雅地处理float的错误，让它返回输入值。我们可以写一个函数，在try/except中调用float


```python
# def attempt_float(x):
#     try:
#         return float(x)
#     except:
#         return x
# 当float(x)抛出异常时，才会执行except的部分：
# 你可能注意到float抛出的异常不仅是ValueError
def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        return x
#     except:(ValueError,TypeError) 可以用元组包含多个异常
attempt_float((1,2))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-233-cd50b16b9c9f> in <module>
         12         return x
         13 #     except:(ValueError,TypeError) 可以用元组包含多个异常
    ---> 14 attempt_float((1,2))
    

    <ipython-input-233-cd50b16b9c9f> in attempt_float(x)
          8 def attempt_float(x):
          9     try:
    ---> 10         return float(x)
         11     except ValueError:
         12         return x


    TypeError: float() argument must be a string or a number, not 'tuple'



```python
f = open(path, 'w')

try:
    write_to_file(f)
except:
    print('Failed')
else:
    print('Succeeded')
finally:
    f.close()
# 你想无论try部分的代码是否成功，都执行一段代码。可以使用finally
# 可以用else让只在try部分成功的情况下，才执行代码
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-234-1d0bf5e92eaa> in <module>
    ----> 1 f = open(path, 'w')
          2 
          3 try:
          4     write_to_file(f)
          5 except:


    NameError: name 'path' is not defined


### 文件和操作系统


```python
# 为了打开一个文件以便读写，可以使用内置的open函数以及一个相对或绝对的文件路径
path = 'examples/segismundo.txt'
f=open(path)
# 默认情况下，文件是以只读模式（'r'）打开的
for line in f:
    print(line)
# 从文件中取出的行都带有完整的行结束符（EOL）用到strip
lines = [x.rstrip() for x in open(path)]
f.close() 记得关闭文件
```


```python
#用with语句可以可以更容易地清理打开的文件
In [212]: with open(path) as f:
    lines = [x.rstrip() for x in f]
#可以在退出代码块时，自动关闭文件
#如果输入f = open(path,'w')，就会有一个新文件被创建在examples/segismundo.txt，
```


```python
# 对于可读文件，一些常用的方法是read、seek和tell。read会从文件返回字符
# 向文件写入，可以使用文件的write或writelines方法
with open('tmp.txt', 'w') as handle:
    handle.writelines(x for x in open(path) if len(x) > 1)
with open('tmp.txt') as f:
    lines = f.readlines()
```
