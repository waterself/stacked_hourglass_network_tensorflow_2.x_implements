import tensorflow as tf
from tensorflow import keras
from keras import layers


# TODO: Using Conv, Max Pooling To DownSampling
# TODO: The Resolution Is 256X1X1 -> 128X3X3 -> 128X1X1
# TODO: Nearest Neighbor Upsampling
'''
PreProcessing
Minimum Resolution is 64X64, Start With 7X7 Filter
Stride 2
Batch Normalization
'''

'''
in paper, fig3 residual block
momentum and epsilon is defualt value
skip operation using add operation
'''
class Residual(keras.layers.Layer):
    def __init__(self,
                 filters=256,
                 momentum=0.99,
                 epsilon=0.001,
                 debugPrint=False,):
        super(Residual, self).__init__()

        self.debugPrint = debugPrint

        self.filters = filters
        
        self.relu = keras.layers.ReLU()
        #print(filters//2)
        # define first - 128x1x1 conv
        self.batchNorm1 = keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv1 = keras.layers.Conv2D(filters=filters//2,kernel_size=1, activation=None, padding='same')
        
        #define second - 128x3x3
        self.batchNorm2 = keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv2 = keras.layers.Conv2D(filters=filters//2,kernel_size=3, activation=None, padding='same')
        
        #define third - 256x1x1
        self.batchNorm3 = keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv3 = keras.layers.Conv2D(filters=filters,kernel_size=1, activation=None, padding='same')

        #define skip - 256x1x1
        self.skipLayer = layers.Conv2D(filters=filters, kernel_size=1, activation=None, padding='same')


    def call(self, inputs, mask=None):
        # if inputs channel same as filter size
        if inputs.shape[-1] == self.filters:
            shortCut = inputs
        else:
            shortCut = self.skipLayer(inputs)
            
        
        x = self.batchNorm1(inputs)
        x = self.relu(x)
        x = self.conv1(x)
        if self.debugPrint == True:
            print("conv1:",x.shape)

        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.debugPrint == True:
            print("conv2",x.shape)

        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.debugPrint == True:
            print("conv3",x.shape)

        x = tf.keras.layers.Add()([x, shortCut])
        return x
    
class PreSequence(keras.layers.Layer):
    def __init__(self, features = 256, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1 = keras.layers.Conv2D(filters=features//4, kernel_size=7, strides=(2, 2), activation='relu', padding='same')
        self.batchNorm1 = keras.layers.BatchNormalization()
        self.residual1 = Residual(features//2)
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))
        self.residual2 = Residual(features//2)
        self.residual3 = Residual(features)

    def call(self, inputs, *args, **kwargs):
        #tf.print("first_input", inputs.shape)
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.residual1(x)
        x = self.pool1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        return x
    

'''
implements for hourglass module,
using recursive definition
'''
class Hourglass(keras.layers.Layer):
    def __init__(
            self, 
            depth,
            features, 
            classes, 
            trainable=True, 
            name=None, dtype=None,
            dynamic=False,
            debugPrint=False,  **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.debugPrint = debugPrint
        self.up1 = Residual(features)

        #Low branch
        self.pool1 = tf.keras.layers.MaxPool2D((2,2))
        self.low1 = Residual(features)
        self.depth = depth


        # Recursive hourglass
        # Resolution will downsampling to 64x64
        if self.depth > 1:
            self.low2 = Hourglass(self.depth-1, features, classes, debugPrint=self.debugPrint)
        else:
            self.low2 = Residual(features)

        self.low3 = Residual(features)
        self.up2 = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")

    def call(self, inputs, *args, **kwargs):
        up1 = self.up1(inputs)
        if self.debugPrint == True:
            print('up1:', up1.shape)

        pool1 = self.pool1(inputs)
        if self.debugPrint == True:
            print('pool1:', pool1.shape)

        low1 = self.low1(pool1)
        if self.debugPrint == True:
            print('low1:', low1.shape)

        low2 = self.low2(low1)
        if self.debugPrint == True:
            print('low2:', low2.shape)

        low3 = self.low3(low2)
        if self.debugPrint == True:
            print('low3:', low2.shape)

        up2 = self.up2(low3)
        if self.debugPrint == True:
            print('up2:', up2.shape)

        return keras.layers.Add()([up1, up2])
    

# class IntermediateBlock(keras.layers.Layer):
#     def __init__(self,
#                  features,
#                  classes,
#                    trainable=True,
#                    name=None,
#                    dtype=None,
#                    dynamic=False,
#                    **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         #relu - linear to feature for nonlinearity
#         self.residual1 = Residual(features)
#         self.next1 = keras.layers.Conv2D(filters=features, kernel_size=1, activation=None, padding='same', )
#         self.batchNorm1 = keras.layers.BatchNormalization()
#         self.relu = keras.layers.ReLU()
#         self.middle1 = keras.layers.Conv2D(filters=classes, kernel_size=1,activation='linear', padding='same')

#     def call(self, inputs, *args, **kwargs):
#         x = self.residual1(inputs)
#         x = self.batchNorm1(x)
#         x = self.next1(x)
#         x = self.batchNorm1(x)
#         x = self.relu(x)
#         heatmap = self.middle1(x)
#         return x, heatmap
    
# class NextBlock(keras.layers.Layer):
#     def __init__(self,
#                 features,
#                 classes,
#                 trainable=True,
#                 name=None,
#                 dtype=None,
#                 dynamic=False,
#                 **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         #relu - linear to feature for nonlinearity
#         self.next2 = keras.layers.Conv2D(filters=features, kernel_size=1, activation='linear',padding='same')
#         self.middle2 = keras.layers.Conv2D(filters=features, kernel_size=1, activation='linear', padding='same')

#     def call(self, inputs, *args, **kwargs):
#         x, heatmap = inputs
#         mid = self.middle2(heatmap)
#         x = self.next2(x)
#         return x, mid


class IntermediateBlock(keras.layers.Layer):
    def __init__(self,
                 features,
                 classes,
                   trainable=True,
                   name=None,
                   dtype=None,
                   dynamic=False,
                   **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        #relu - linear to feature for nonlinearity
        self.next1 = keras.layers.Conv2D(filters=features, kernel_size=1, activation='relu', padding='same')
        self.batchNorm1 = keras.layers.BatchNormalization()
        self.middle1 = keras.layers.Conv2D(filters=classes, kernel_size=1,activation='linear',  padding='same')
        self.next2 = keras.layers.Conv2D(filters=features, kernel_size=1, activation='linear',padding='same')
        self.middle2 = keras.layers.Conv2D(filters=features, kernel_size=1, activation='linear', padding='same')

    def call(self, inputs, *args, **kwargs):
        x = self.next1(inputs)
        x = self.batchNorm1(x)
        heatmap = self.middle1(x)
        mid = self.middle2(heatmap)
        x = self.next2(x)
        return x, mid, heatmap
    
    def get_next_weight(self):
        return self.middle2.trainable_variables, self.next2.trainable_variables



'''
linked class for supervision
'''
class HourglassWithSuperVision(keras.layers.Layer):
    def __init__(self, classes, depth, features=256, supervision=True , trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        self.Hourglass = Hourglass(depth=depth, features=features, classes=classes)
        self.SuperVision = IntermediateBlock(features, classes)

    def call(self, inputs, *args, **kwargs):
        x = self.Hourglass(inputs)
        x, mid, y = self.SuperVision(x)
        return keras.layers.Add()([x, mid, inputs]), y
    
# '''
# linked class for supervision
# '''
# class HourglassWithSuperVision(keras.layers.Layer):
#     def __init__(self, classes, depth, features=256, supervision=True , trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         self.Hourglass = Hourglass(depth=depth, features=features, classes=classes)
#         self.SuperVision = IntermediateBlock(features, classes)

#     def call(self, inputs, *args, **kwargs):
#         x = self.Hourglass(inputs)
#         x, mid, y = self.SuperVision(x)
#         return keras.layers.Add()([x, mid, inputs]), y




