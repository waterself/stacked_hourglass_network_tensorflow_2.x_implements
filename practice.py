
# def CalcOutput(input_size, filter_size, stride, padding):
#     return (input_size - filter_size +  padding * 2 // stride) + 1

# res = CalcOutput(32, 5, 2, 0)
# print(res)

import tensorflow as tf
from src.layers import layers

x = tf.random.uniform(shape=(2,256,256,3), minval=0, maxval=255, dtype=tf.float32)

hourglass = layers.Hourglass(depth=2, classes=16, features=256, debugPrint=True)
output = hourglass(x)
#output = layers.Residual(256, debugPrint=True)(x)

print(output.shape)

# def loop_test(inputs = 0, stacks = 8):
#     x = inputs + 1

#     for idx in range(1, stacks - 1):
#         x = x + 1

#     x = x + 1
    
#     return x

# res = loop_test()
# print(res)
