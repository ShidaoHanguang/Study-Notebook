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