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

回归问题比起分类问题，预测的是一个连续值而不是离散的标签，因此对应损失函数要进行修改。

##### 3.5.1 导入数据集

```python
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
#数据标准化，对输入值的每个特征，先减去特征平均值，再除以标准差，得到的平均值平均值为0，标准差为1，实现标准化
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
```

##### 3.5.2 构建网络

```python
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #使用mse损失函数，即均方误差，预测值与目标值之差的平方
    #同时使用的指标为mae平均绝对误差，指的是预测值与目标值之差的绝对值
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
```

##### 3.5.3 K折验证

在训练集数据较少的情况下，如果想要划分出训练集和验证集，可以参考使用**K折交叉验证**。这种方法将可用数据划分为 K个分区（K 通常取 4 或 5），实例化 K 个相同的模型，将每个模型在 K-1 个分区上训练，并在剩下的一个分区上进行评估。模型的验证分数等于 K 个验证分数的平均值。

![image-20240331205034369](/home/shidaohg/WorkPlace/Git/Study-Notebook/深度学习/material/3_2.png)

```python
import numpy as np
k = 4
#地板除法，除后只保留整数部分
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
#进行k次验证
for i in range(k):
    print('processing fold #', i)
    #划分验证集
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    #剩下的划分训练集
    partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples],
                train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate(
        		[train_targets[:i * num_val_samples],
        		train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    #开始训练，epochs表示取样次数，batch_size表示一次取样量，verbose取0表示静默模式
    model.fit(partial_train_data, partial_train_targets,
        epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
```

之后，可以考虑增大取样次数，增大训练量

```python
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
```

##### 3.5.4 绘制图像

```python
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
```

```python
#绘制平滑曲线，factor参数控制平滑的程度
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            #计算当前数据点与上一个数据点的加权平均
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

#删除前10个点
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
```

### 4 机器学习基础

#### 4.1 机器学习的四个分支

##### 4.1.1 监督学习

监督学习指给定一组样本（通常由人工标注），学会将输入数据映射到已知目标（也叫**标注**）。

监督学习主要包括各种分类和回归，同时还有很多的奇特变体，主要包括以下：

- **序列生成**（sequence generation）：给定一张图像，预测描述图像的文字。序列生成有时可以被重新表示为一系列分类问题，比如反复预测序列中的单词或标记。
- **语法树预测**（syntax tree prediction）：给定一个句子，预测其分解生成的语法树。
- **目标检测**（object detection）：给定一张图像，在图中特定目标的周围画一个边界框。这个问题也可以表示为分类问题（给定多个候选边界框，对每个框内的目标进行分类）或分类与回归联合问题（用向量回归来预测边界框的坐标）。
- **图像分割**（image segmentation）：给定一张图像，在特定物体上画一个像素级的掩模（mask）。

##### 4.1.2 无监督学习

**无监督学习**是指在没有目标的情况下寻找输入数据的变换，其目的在于数据可视化、数据压缩、数据去噪或更好地理解数据中的相关性。无监督学习是数据分析的必备技能，在解决监督学习问题之前，为了更好地了解数据集，它通常是一个必要步骤。**降维**（dimensionalityreduction）和**聚类**（clustering）都是众所周知的无监督学习方法。

##### 4.1.3 自监督学习

自监督学习是没有人工标注的标签的监督学习，标签从输入数据中自动生成，通常是通过启发式算法获得。

**自编码器**就是一种自监督学习的例子，其生成的目标就是未经修改的输入。此外，通过给定视频中过去的帧预测下一帧，或者给定文本中前面的词来预测下一个词，也都是自监督学习的例子（也可以划分为**时序监督学习**）。

##### 4.1.4 强化学习

在强化学习中，**智能体**（agent）接收有关其环境的信息，并学会选择使某种奖励最大化的行动。例如，神经网络会“观察”视频游戏的屏幕并输出游戏操作，目的是尽可能得高分，这种神经网络可以通过强化学习来训练。

##### 4.1.5 分类和回归的术语表

- **样本**（sample）或**输入**（input）：进入模型的数据点。
- **预测**（prediction）或输出（output）：从模型出来的结果。
- **目标**（target）：真实值。对于外部数据源，理想情况下，模型应该能够预测出目标。
- **预测误差**（prediction error）或**损失值**（loss value）：模型预测与目标之间的距离。
- **类别**（class）：分类问题中供选择的一组标签。例如，对猫狗图像进行分类时，“狗”和“猫”就是两个类别。
- **标签**（label）：分类问题中类别标注的具体例子。比如，如果 1234 号图像被标注为包含类别“狗”，那么“狗”就是 1234 号图像的标签。
- **真值**（ground-truth）或**标注**（annotation）：数据集的所有目标，通常由人工收集。
- **二分类**（binary classification）：一种分类任务，每个输入样本都应被划分到两个互斥的类别中。
- **多分类**（multiclass classification）：一种分类任务，每个输入样本都应被划分到两个以上的类别中，比如手写数字分类。
- **多标签分类**（multilabel classification）：一种分类任务，每个输入样本都可以分配多个标签。举个例子，如果一幅图像里可能既有猫又有狗，那么应该同时标注“猫”标签和“狗”标签。每幅图像的标签个数通常是可变的。
- **标量回归**（scalar regression）：目标是连续标量值的任务。预测房价就是一个很好的例子，不同的目标价格形成一个连续的空间。
- **向量回归**（vector regression）：目标是一组连续值（比如一个连续向量）的任务。如果对多个值（比如图像边界框的坐标）进行回归，那就是向量回归。
- **小批量**（mini-batch）或批量（batch）：模型同时处理的一小部分样本（样本数通常为 8~128）。样本数通常取 2 的幂，这样便于 GPU 上的内存分配。训练时，小批量用来为模型权重计算一次梯度下降更新。



#### 4.2 评估

在例子中，经过几轮训练，模型都很快出现**过拟合**，其在训练数据上的性能始终在提高，但在前所未见的数据上的性能则有所下降。机器学习的目的是得到泛化的模型，需要通过衡量模型泛化能力，进一步降低过拟合带来的负面影响。

##### 4.2.1 数据划分

评估模型重点在于将数据划分为三个集合：训练集、验证集和测试集。之所以不单单使用训练集和测试集，是因为在开发模型时对模型配置需要通过在验证集上的性能作为调节信号，用来测验模型的过拟合情况。常见的评估方法有以下几种：

###### 1. 简单的留出验证

留出一定比例的数据作为测试集，并在剩余的数据上训练模型，最后在测试集上评估模型。为了防止信息泄露，不能基于测试集来调节模型，还应保留一个验证集。

###### 2. K折验证

**K 折验证**（K-fold validation）将数据划分为大小相同的 K 个分区。对于每个分区 i，在剩余的 K-1 个分区上训练模型，然后在分区 i 上评估模型。最终分数等于 K 个分数的平均值。对于不同的训练集 - 测试集划分，如果模型性能的变化很大，那么这种方法很有用。与留出验证一样，这种方法也需要独立的验证集进行模型校正。

###### 3. 带有打乱数据的重复 K 折验证

如果可用的数据相对较少，又需要尽可能精确地评估模型，可以选择带有打乱数据的重复 K 折验证（iterated K-fold validation with shuffling）。具体做法是多次使用 K 折验证，在每次将数据划分为 K 个分区之前都先将数据打乱。最终分数是每次 K 折验证分数的平均值。注意，这种方法一共要训练和评估 P×K 个模型（P是重复次数），计算代价很大。

##### 4.2.2 评估的注意事项

选择模型评估方法时，需要注意以下几点。

- **数据代表性**（data representativeness）：训练集和测试集都能够代表当前数据。例如，你想要对数字图像进行分类，而图像样本是按类别排序的，如果你将前 80% 作为训练集，剩余 20% 作为测试集，那么会导致训练集中只包含类别 0~7，而测试集中只包含类别 8~9。因此，在将数据划分为训练集和测试集之前，通常应该**随机打乱**数据。

- **时间箭头**（the arrow of time）：如果想要根据过去预测未来（比如明天的天气、股票走势等），那么在划分数据前你**不**应该随机打乱数据，因为这么做会造成**时间泄露**（temporalleak）。在这种情况下，应该始终确保测试集中所有数据的时间都晚于训练集数据。
- **数据冗余**（redundancy in your data）：如果数据中的某些数据点出现了两次（这在现实中的数据里十分常见），那么打乱数据并划分成训练集和验证集会导致训练集和验证集之间的数据冗余。一定要确保训练集和验证集之间没有交集。

#### 4.3 数据预处理、特征工程和特征学习

#### 4.3.1 数据预处理

数据预处理的目的是使原始数据更适于用神经网络处理，包括向量化、标准化、处理缺失值和特征提取。

1. 向量化：神经网络的所有输入和目标都必须是浮点数张量（在特定情况下可以是整数张量）。无论处理什么数据（声音、图像还是文本），都必须首先将其转换为张量，这一步叫作**数据向量化**（data vectorization）。

2. 值标准化：一般来说，输入数据应该具有以下特征。

   - 取值较小：大部分值都应该在 0~1 范围内。

   - 同质性（homogenous）：所有特征的取值都应该在大致相同的范围内。

   - 此外，还有较为严格的标准化方法：将每个特征分别标准化，使其平均值为 0，标准差为 1。

     ```python
     x -= x.mean(axis=0)
     x /= x.std(axis=0)
     ```

3. 缺值处理：一般来说，对于神经网络，将缺失值设置为 0 是安全的，只要 0 不是一个有意义的值。网络能够从数据中学到 0 意味着缺失数据，并且会忽略这个值。

   

##### 4.3.2 特征工程

特征工程（feature engineering）是指将数据输入模型之前，利用关于数据和机器学习算法（这里指神经网络）的知识对数据进行硬编码的变换（不是模型学到的），以改善模型的效果。多数情况下，一个机器学习模型无法从完全任意的数据中进行学习。呈现给模型的数据
应该便于模型进行学习。

#### 4.4 过拟合和欠拟合

机器学习的根本问题是优化和泛化之间的对立。**优化**（optimization）是指调节模型以在训练数据上得到最佳性能，而泛化（generalization）是指训练好的模型在前所未见的数据上的性能好坏。

训练开始时，优化和泛化是相关的：训练数据上的损失越小，测试数据上的损失也越小。这时的模型是**欠拟合**（underfit）的，即仍有改进的空间，网络还没有对训练数据中所有相关模式建模。但在训练数据上迭代一定次数之后，泛化不再提高，验证指标先是不变，然后开始变差，即模型开始**过拟合**。

这种情况下，最优解决方法是**获取更多的训练数据**。如果无法获得更多的数据，次优方法是调节模型允许存储的信息量，或者对模型允许存储的信息加以约束。这种方法叫做**正则化**。以下是几种常见的正则化方法：

##### 4.4.1 减小网络大小

防止过拟合的最简单的方法就是**减小模型大小**，即**减少模型中可学习参数的个数**（这由层数和每层的单元个数决定）。在深度学习中，模型中可学习参数的个数通常被称为模型的**容量**（capacity）。直观上来看，参数更多的模型拥有更大的记忆容量（memorization capacity），因此能够在训练样本和目标之间轻松地学会完美的字典式映射，这种映射没有任何泛化能力。

##### 4.4.2 添加权重正则化

简单模型比复杂模型更不容易过拟合。

这里的**简单模型**（simple model）是指参数值分布的熵更小的模型（或参数更少的模型）。因此，一种常见的降低过拟合的方法就是强制让模型权重只能取较小的值，从而限制模型的复杂度，这使得权重值的分布更加**规则**（regular）。这种方法叫作**权重正则化**
（weight regularization），其实现方法是向网络损失函数中添加与较大权重值相关的**成本**（cost）。
这个成本有两种形式。

- L1 正则化（L1 regularization）：添加的成本与权重系数的绝对值［权重的 L1 范数（norm）］成正比。
- L2 正则化（L2 regularization）：添加的成本与权重系数的平方（权重的 L2 范数）成正比。神经网络的 L2 正则化也叫**权重衰减**（weight decay）。

在 Keras 中， 添加权重正则化的方法是向层传递**权重正则化项实例**（weight regularizerinstance）作为关键字参数。

```python
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

$l2(0.001)$ 的意思是该层权重矩阵的每个系数都会使网络总损失增加 $0.001 * weight\_coefficient\_value$。

##### 4.4.3 添加dropout正则化

对某一层使用 dropout，就是在训练过程中随机将该层的一些输出特征**舍弃**（设置为 0）。**dropout 比率**（dropout rate）是被设为 0 的特征所占的比例，通常在 0.2~0.5范围内。测试时没有单元被舍弃，而该层的输出值需要按 dropout 比率**缩小**，因为这时比训练时有更多的单元被激活，需要加以平衡。

```python
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
#测试时
layer_output *= 0.5
```

也可以在训练时就提前进行运算，使得测试时输出保持不变。

```python
layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
layer_output /= 0.5
```



#### 4.5 机器学习通用工作流程

##### 4.5.1 定义问题，收集数据集

确定输入数据和输出数据的类型，确定所面对的问题的类型。

特殊情况：面对**非平稳问题**（nonstationary problem）时，需要不断地利用最新数据重新训练模型，或者在一个问题是平稳的时间尺度上收集数据。

##### 4.5.2 选择衡量成功的指标

需要明确成功的定义：

- 平衡分类问题：精度和**接收者操作特征曲线下面积**（area under the receiver operating characteristic curve，ROC AUC）是常用的指标。
- 类别不平衡问题：准确率（precision）和召回率（recall）。
- 排序问题或多标签分类问题：平均准确率均值（mean average precision）

##### 4.5.3 确定评估方法

上文提到过三种常见的评估方法：

- **留出验证集**：数据量很大时可以采用这种方法。
- **K 折交叉验证**：如果留出验证的样本量太少，无法保证可靠性，那么应该选择这种方法。
- **重复的 K 折验证**：如果可用的数据很少，同时模型评估又需要非常准确，那么应该使用
  这种方法。

##### 4.5.4 准备数据

将输入数据格式转化为张量，这些张量的取值通常应该缩放为较小的值，比如在[-1，1]或者[0，1]区间内。如果不同特征取值范围不同，应该做标准化。

##### 4.5.5 开发模型

目的是获得**统计功效**（statistical power），通过开发一个小型模型，能够打败纯随机的**基准**（dumb baseline）。

此外，还需要选择三个关键参数来构建工作模型：

- **最后一层的激活**：它对网络输出进行有效的限制。例如，IMDB 分类的例子在最后一层使用了 sigmoid，回归的例子在最后一层没有使用激活，等等。
- **损失函数**：它应该匹配你要解决的问题的类型。例如，IMDB 的例子使用 binary_crossentropy、回归的例子使用 mse，等等。
- **优化配置**：你要使用哪种优化器？学习率是多少？大多数情况下，使用 rmsprop 及其
  默认的学习率是稳妥的。

|      问题类型      | 最后一层激活 |         损失函数         |
| :----------------: | :----------: | :----------------------: |
|     二分类问题     |   sigmoid    |    binary_crossentroy    |
| 多分类、单标签问题 |   softmax    | categorical_crossentropy |
| 多分类、多标签问题 |   sigmoid    |   binary_crossentropy    |
|    回归到任意值    |      无      |           mse            |
|   回归到标准化值   |   sigmoid    | mse或binary_crossentroy  |

##### 4.5.6 扩大模型规模

通过不断扩大模型的规模，探索过拟合的模型，并在此基础上进行正则化和调节模型，使得结果模型既不欠拟合也不过拟合。

##### 4.5.7 正则化与调节超参数

不断地调节模型、训练、在验证数据上评估（这里不是测试数据）、再次调节模型，然后重复这一过程，直到模型达到最佳性能。

- 添加 dropout。
- 尝试不同的架构：增加或减少层数。
- 添加 L1 和 / 或 L2 正则化。
- 尝试不同的超参数（比如每层的单元个数或优化器的学习率），以找到最佳配置。
- （可选）反复做特征工程：添加新特征或删除没有信息量的特征。
