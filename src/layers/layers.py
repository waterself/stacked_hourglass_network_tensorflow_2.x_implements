import tensorflow as tf
from tensorflow import keras


# TODO: Using Conv, Max Pooling To DownSampling
# TODO: The Resolution Is 256X1X1 -> 128X3X3 -> 128X1X1
# TODO: Nearest Neighbor Upsampling
'''
Minimum Resolution is 64X64, Start With 7X7 Filter
Stride 2
Batch Normalization
'''

# TODO: Input Image - Conv Operation -> Max Pooling -> Conv Operation -> Max Pooling -> 
# class ConvNet(keras.layers.Layer):
#     '''
#     need unit(output_size), input_dim = input_size
#     '''
#     def __init__(self, unit, input_dim, channel):
#         super(ConvNet, self).__init__()
#         w_init = tf.random_normal_initializer()

class Residual(keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        conv1 = keras.layers.Conv2D(input_dim, )





    

class Hourglass(keras.layers.Layer):
    pass