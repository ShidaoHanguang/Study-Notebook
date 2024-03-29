import tensorflow as tf

print('GPU',tf.test.is_gpu_available())

a = tf.constant(2.)
b = tf.constant(4.)

print(a * b)
