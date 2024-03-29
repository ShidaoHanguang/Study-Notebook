# 深度学习

本文基于对《基于python深度学习实战》一书的学习，以下内容大部分出自该书。

## 一、基础概念

### 1. 机器学习简述

#### 1.1 概率建模



#### 1.2 早期神经网络



#### 1.3 核方法



#### 1.4 决策树、随机森林与梯度提升机



### 2. 神经网络的数学基础

#### 2.1 组成

分类问题中某个类别被称作**类**（class）；数据点叫做**样本**（sample）；某个样本对应的类叫做**标签**（label）。

模型通过对**训练集**的学习后，对**测试集**进行测试，验证模型的准确性和性能。

神经网络的核心组件是**层**（layer），这是一种数据处理模块，通过对输入数据进行过滤筛选，提取其中更需要的内容。通过将层进行链接，从而实现渐进式的**数据蒸馏**。深度学习的模型就包含一系列逐步精细的层。

要想训练网络，我们还需要选择**编译**步骤的三个参数。

- **损失函数**（loss function）：网络如何衡量在训练数据上的性能，即网络如何朝着正确的方向前进
- **优化器**（optimizer）：基于训练数据和损失函数来更新网络的机制
- **指标**（metric）：精度等（在训练和测试过程中需要监控的参数）



#### 2.2 代码演示

```python
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
# 准备训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 引入神经网络模型
network = models.Sequential()
# 添加2个密集连接（也称全连接，Dense）层
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 10路softmax层,将会返回一个由10个概率值(总和为1)组成的数组
network.add(layers.Dense(10, activation='softmax'))
# 实现编译功能
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 数据集预处理,将(60000,28,28)转为(60000, 28 * 28),并且归一化
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 对标签进行分类编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 通过训练数据集进行模型拟合(fit)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 利用测试集进行性能验证
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
```

**代码解释**：通过引入神经网络模型，利用训练集对模型进行训练，再对测试集进行测试，通过精度反馈模型效果。

**注意事项**：本示例中训练集训练后精度为98%，但测试后精度为97.8%左右，这种训练精度和测试精度的差距被称作**过拟合**。



#### 2.3 神经网络的数据表示

数据存储在多维Numpy数组中，叫做**张量**（tensor）,这是机器学习系统中基本的数据结构。张量的**维度**（dimension）通常被称作**轴**（axis）。

##### 2.3.1 标量（0D张量）

仅包含一个数字的张量叫作**标量**（scalar，也叫标量张量、零维张量、0D 张量）。

在numpy中，一个 float32 或 float64 的数字就是一个标量张量（或标量数组）。可以用 **ndim** 属性查看一个张量的轴的个数，张量轴的个数也叫做**阶**（rank）。

```python
import numpy as np
x = np.array(12)
print(x.ndim)
```

运行后结果为0

##### 2.3.2 向量（1D张量）

数字组成的数组叫做**向量**（vector，也叫作一维张量，1D张量），只有一个轴。

一个向量中包含n个元素，可作**nD向量**，注意不要同nD张量搞混。

##### 2.3.3 矩阵（2D张量）

向量组成的数组称作**矩阵**（matrix，二维张量，2D张量）。矩阵有两个轴（通常被称作**行**和**列**）。

##### 2.3.4 多阶张量

多个矩阵的组合可以形成多阶的张量

##### 2.3.5 关键属性

- **轴的个数（阶）**	可以用np中的ndim属性查看。
- **形状**	一个整数元组，表示张量沿每个轴的维度大小（个数）。
- **数据类型**  	张量所包含数据的类型，一般是uint8、float32、float64等。



#### 2.4 张量运算

##### 2.4.1 逐元素运算

包括逐元素的**加法**和**relu算法**，在numpy中可以用代码快速实现

```python
import numpy as np

z = x + y
#relu算法是一种滤值方法，通过比较元素同0的大小，保留大于0的部分
z = np.maximum(z, 0.)
```



##### 2.4.2 广播

如果将两个形状不同的张量相加时，没有歧义的话，较小的张量会被**广播**（broadcast），以匹配较大张量的形状。

广播包含以下两步:

1. 向较小的张量添加轴（叫作广播轴），使其 ndim 与较大的张量相同。
2. 将较小的张量沿着新轴重复，使其形状与较大的张量相同。



##### 2.4.3 张量点积

点积运算，也叫**张量积**，是最常见也最有用的张量运算。与逐元素的运算不同，它将输入张量的元素合并在一起。

![张量点积](/home/shidaohg/WorkPlace/Git/Study-Notebook/深度学习/material/2.1.png)



##### 2.4.4 张量变形

张量变形是指改变张量的行和列，以得到想要的形状。变形后的张量的元素总个数与初始张量相同。

经常遇到的一种特殊的张量变形是转置（transposition）。对矩阵做转置是指将行和列互换，使 x[i, :] 变为 x[:, i]。



#### 2.5 基于梯度的优化

##### 2.5.1 概念

下面的式子展示了一种简单的神经层运算方法：

$$output = relu(dot(W, input) + b)$$

其中，W 和 b 都是张量，均为该层的属性。它们被称为该层的**权重**（weight，或者**可训练参数**，trainable parameter），这些权重包含网络从观察

训练数据中学到的信息。

一开始，这些权重矩阵取较小的随机值，这一步叫作**随机初始化**（random initialization）。当然，W 和 b 都是随机的，relu(dot(W, input) + b) 肯定不会得到任何有用的表示。虽然得到的表示是没有意义的，但这是一个起点。下一步则是根据反馈信号逐渐调节这些权重。这个逐渐调节的过程叫作**训练**，也就是机器学习中的学习。上述过程发生在一个**训练循环**（training loop）内，其具体过程如下。必要时一直重复这些步骤。

1.  抽取训练样本 x 和对应目标 y 组成的数据批量。
2.  在 x 上运行网络［这一步叫作**前向传播**（forward pass）］，得到预测值 y_pred。
3.  计算网络在这批数据上的损失，用于衡量 y_pred 和 y 之间的距离。
4.  更新网络的所有权重，使网络在这批数据上的损失略微下降。

最终得到的网络在训练数据上的损失非常小，即预测值 y_pred 和预期目标 y 之间的距离非常小。

其中，更新网络权重的步骤是较为困难的，很多时候不知道如何变动各项权重，盲目改变后重新计算是非常低效的，为此，一种更好的方法是利用网络中所有运算都是**可微**（differentiable）的这一事实，计算损失相对于网络系数的**梯度**（gradient），然后向梯度的反方向改变系数，从而使损失降低。

梯度是张量运算的导数。它是导数这一概念向多元函数导数的推广。多元函数是以张量作为输入的函数。

##### 2.5.2 随机梯度下降

给定一个可微函数，理论上可以用解析法找到它的最小值：函数的最小值是导数为 0 的点，因此你只需找到所有导数为 0 的点，然后计算函数在其中哪个点具有最小值。

由于处理的是一个可微函数，可以计算出它的梯度，从而有效地实现第四步。沿着梯度的反方向更新权重，损失每次都会变小一点。

1.  抽取训练样本 x 和对应目标 y 组成的数据批量。
2.  在 x 上运行网络，得到预测值 y_pred。
3.  计算网络在这批数据上的损失，用于衡量 y_pred 和 y 之间的距离。
4.  计算损失相对于网络参数的梯度［一次**反向传播**（backward pass）］。
5.  将参数沿着梯度的反方向移动一点，比如 W -= step * gradient，从而使这批数据上的损失减小一点。

这种方法叫作**小批量随机梯度下降**（mini-batch stochastic gradient descent，又称为**小批量 SGD**）

每次迭代时只抽取一个样本和目标，而不是抽取一批数据，这叫作**真 SGD**。

每一次迭代都在所有数据上运行，这叫作**批量 SGD**。

此外，SGD 还有多种变体，其区别在于计算下一次权重更新时还要考虑上一次权重更新，而不是仅仅考虑当前梯度值，比如带动量的 **SGD**、**Adagrad**、**RMSProp** 等变体。这些变体被称为**优化方法**（optimization method）或**优化器**（optimizer）。

动量解决了 SGD 的两个问题：收敛速度和局部极小点。动量方法的实现过程是每一步都移动小球，不仅要考虑当前的斜率值（当前的加速度），还要考虑当前的速度（来自于之前的加速度）。这在实践中的是指，更新参数 w 不仅要考虑当前的梯度值，还要考虑上一次的参数更新。

##### 2.5.3 反向传播算法

根据微积分的知识，这种函数链可以利用下面这个恒等式进行求导，它称为**链式法则**（chain rule）：$(f(g(x)))' = f'(g(x)) * g'(x)$​。将链式法则应用于神经网络梯度值的计算，得到的算法叫作**反向传播**（backpropagation，有时也叫反式微分，reverse-mode differentiation）。反向传播从最终损失值开始，从最顶层反向作用至最底层，利用链式法则计算每个参数对损失值的贡献大小。

### 3. 神经网络入门

#### 3.1神经网络深层剖析

##### 3.1.1 层

神经网络的基本数据结构是**层**。层是一个数据处理模块，将一个或多个输入张量转换为一个或多个输出张量。有些层是无状态的，但大多数的层是有状态的，即层的**权重**。权重是利用随机梯度下降学到的一个或多个张量，其中包含网络的**知识**。

通常，简单的向量数据保存在形状为（samples, features）的2D张量中，用**密集连接层**［densely connected layer，也叫**全连接层**（fully connected layer）或**密集层**（dense layer），对应于 Keras 的 Dense 类］来处理。

序列数据保存在形状为 (samples, timesteps, features) 的 3D 张量中，通常用**循环层**（recurrent layer，比如 Keras 的 LSTM 层）来处理。

图像数据保存在 4D 张量中，通常用二维**卷积层**（Keras 的 Conv2D）来处理。

```python
from keras import models
from keras import layers
model = models.Sequential()
# 创建一个层，只接受第一个维度大小是784的2D张量，返回一个第一个维度大小为32的张量
model.add(layers.Dense(32, input_shape=(784,)))
# 该层只能接受一个32维的向量
model.add(layers.Dense(32))
```

##### 3.1.2 模型

深度学习模型是层构成的**有向无环图**。

最常见的例子就是层的线性堆叠，将单一输入映射为单一输出，除此以外还有一些常见的网络拓扑结构：

- 双分支（two-branch）网络
- 多头（multihead）网络
- Inception 模块

网络的拓扑结构定义了一个**假设空间**（hypothesis space）。选定了网络拓扑结构，意味着将假设空间限定为一系列特定的张量运算，将输入数据映射为输出数据，需要为这些张量运算的权重张量找到一组合适的值。

##### 3.1.3 损失函数与优化器

**损失函数**（**目标函数**）：在训练过程中需要将其最小化，用于衡量当前任务是否已完成。

**优化器**：决定如何基于损失函数对网络进行更新，通常执行随机梯度下降（SGD）的某个变体。

具有多个输出的神经网络可能具有多个损失函数（每个输出对应一个损失函数）。但是，梯度下降过程必须基于**单个**标量损失值。因此，对于具有多个损失函数的网络，需要将所有损失函数**取平均**，变为一个标量值。

一些简单的关于损失函数选择方面的指导原则：

- 二分类问题：二元交叉熵（binary crossentropy）损失函数
- 多分类问题：分类交叉熵（categorical crossentropy）损失函数
- 回归问题：均方误差（mean-squared error）损失函数
- 序列学习问题：联结主义时序分类（CTC，connectionist temporal classification）损失函数

#### 3.2 二分类问题

本节使用 IMDB 数据集，它包含来自互联网电影数据库（IMDB）的 50 000 条严重两极分化的评论。数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试集都包含 50% 的正面评论和 50% 的负面评论。

##### 3.2.1 导入数据集

```python
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

**代码解释**：train_data与test_data是组成是各个评论，其中每种评论都已经转化成了由数字组成的列表，数字表示的是该字符的出现频率，当前示例中截取了到10000的字符，也就是说列表中的数字出现大小限定在[1，9999]之间。

train_labels与test_labels代表其评论的状态，其中0表示负面，1表示正面。

之后根据评论值解码相应英文单词

```python
# 获取对应字符字典
word_index = imdb.get_word_index()
# 将键值对调
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
# 对第一句话进行解码，其中-3是因为索引0-2分别对应了一些特殊情况
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
```

**代码解释**：因为原来字符字典是单词对应频率的格式，而data中存储的都是频率数字，因此需要通过键值对换，用于快速根据数字查询对应字符。

##### 3.2.2 数据集处理

```python
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # 将数据集整合到一个2D张量内
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
```

##### 3.2.3 构建网络

###### 定义模型

```python
from keras import models
from keras import layers
# 定义模型的各层
model = models.Sequential()
# 全连接层的参数（16）是该层隐藏单元的个数，一个隐藏单元（hidden unit）是该层表示空间的一个维度。
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

###### 配置器优化

```python
from keras import optimizers
# optimizers是一种优化器
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
				loss='binary_crossentropy',
				metrics=['accuracy'])
```

###### 使用自定义的损失和指标

```python
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
				loss=losses.binary_crossentropy,
				metrics=[metrics.binary_accuracy])
```

###### 训练模型

```python
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['acc'])
#x_val，y_val是用于验证的验证集，监控模型在前所未见的数据上的精度
history = model.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val, y_val))
```

###### 绘制损失图像

```python
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
# ‘bo’表示蓝色圆点
plt.plot(epochs, loss_values, 'bo', label='Training loss')
# 'b'表示蓝色实线
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# 绘制训练及验证损失
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

```python
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
# 绘制训练及验证精度
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

其图像结果如下所示：

![image-20240329155046940](/home/shidaohg/WorkPlace/Git/Study-Notebook/深度学习/material/3_1.png)

**注意事项**：从验证集精度看，其实第四轮训练时已经有了较好的精度，但在之后的训练中对于验证集的检测精度在波动降低，这其实就是**过拟合**。对于训练数据的过度优化，反而会使得模型更贴合训练数据，无法泛化到训练集之外的数据。

###### 重新开始训练模型

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
				loss='binary_crossentropy',
				metrics=['accuracy'])
# 这次只进行4轮训练
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
```



#### 3.4 多分类问题

```python
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

len(train_data)

len(test_data)

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

one_hot_train_labels.shape

from tensorflow.keras.utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

one_hot_train_labels.shape

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=16,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array)) / len(test_labels)

predictions = model.predict(x_test)
```

大部分情况同二分类问题类似，但还是存在一些注意事项：

- 如果要对 N 个**类别**的数据点进行分类，网络的最后一层应该是大小为 N 的 Dense 层。

- 对于单标签、多分类问题，网络的最后一层应该使用 softmax 激活，这样可以输出在 N个输出类别上的**概率分布**。

- 这种问题的损失函数几乎总是应该使用分类交叉熵。它将网络输出的概率分布与目标的真实分布之间的距离最小化。

- 处理多分类问题的标签有两种方法：

  - 通过分类编码（ 也叫one-hot编码）对标签进行编码，然后使用categorical_crossentropy作为损失函数。

  - 将标签编码为整数，然后使用 sparse_categorical_crossentropy 损失函数。

#### 3.5 回归问题
