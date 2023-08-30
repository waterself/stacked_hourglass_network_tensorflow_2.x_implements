import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer


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
    def __init__(self, kernel_size, filters):
        super(Residual, self).__init__()
        self.relu = tf.nn.relu()
        self.batchNorm1 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv2D( )
        self.batchNorm2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D( )
        self.batchNorm3 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D( )


    def call(self, inputs, training=None, mask=None):
        
        x = self.batchNorm1(inputs)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.conv3(x)
        
        return super().call(inputs, training, mask)







    

class Hourglass(keras.layers.Layer):
    pass