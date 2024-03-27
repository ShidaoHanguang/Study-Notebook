from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
# 准备训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]
print(digit.shape)
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()