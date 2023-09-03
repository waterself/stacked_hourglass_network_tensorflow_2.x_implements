import tensorflow as tf
from tensorflow import keras
from keras import layers


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
    def __init__(self, 
                 momentum=0.99,
                 epsilon=0.001,
                 filters=256):
        super(Residual, self).__init__()
        self.convStride = 2
        self.convKernel = (7,7)
        
        self.relu = keras.layers.ReLU()
        self.batchNorm1 = keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv1 = keras.layers.Conv2D(filters=filters//2,kernel_size=1)
        self.batchNorm2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=filters//2,kernel_size=3)
        self.batchNorm3 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(filters=1,kernel_size=1)
        self.skipLayer = layers.Conv2D(filters=filters, kernel_size=1)


    def call(self, inputs, training=None, mask=None):
        if self.need_skip:
            shortCut = self.skipLayer(inputs)
        else:
            shortCut = self.inputs

        x = self.batchNorm1(inputs)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.conv3(x)
        
        x = tf.add()([x, shortCut])
        return super().call(inputs, training, mask)



class Hourglass(keras.layers.Layer):
    def __init__(self, d, f, batchNorm=None, increase = 0, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        df = 0

        self.up1 = Residual(f)

        #Low branch
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))
        self.low1 = Residual(f)
        self.d = d

        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(d-1, df, batchNorm=batchNorm)
        else:
            self.low2 = Residual(df, df)

        self.low3 = Residual(df, f)
        self.up2 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")

    def call(self, inputs, *args, **kwargs):
        super().call(inputs, *args, **kwargs)
        up1 = self.up1(inputs)
        pool1 = self.pool1(inputs)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return tf.add()([up1, up2])
    