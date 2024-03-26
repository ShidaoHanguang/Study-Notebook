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



