from keras import models
from keras import layers
model = models.Sequential()
# 创建一个层，只接受第一个维度大小是784的2D张量，返回一个第一个维度大小为32的张量
model.add(layers.Dense(32, input_shape=(784,)))
# 该层只能接受一个32维的向量
model.add(layers.Dense(32))