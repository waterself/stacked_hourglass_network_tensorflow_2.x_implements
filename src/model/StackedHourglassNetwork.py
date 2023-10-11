from ..layers import layers
from tensorflow import keras
import tensorflow as tf



class StackedHourglassNet(keras.models.Model):
    def __init__(self, classes, depth=2, features=256, stacks=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacks = stacks
        self.hourglasses = [layers.HourglassWithSuperVision(classes) for idx in range(stacks)]
        self.intermediateOutput = []
        self.preSequence = keras.layers.Conv2D(filters=features,kernel_size=1, activation='relu')

        print('hg_stacks:', len(self.hourglasses) )
    
    def train_step(self, data):
        inputs, target = data
        # TODO: Each Modules DO NOT share weights while learning
        # TODO: Each Modules will learn with their SuperVision
        # TODO: Each Modules use SAME Grounds Truth Target
        '''
            nxt is forward features
            mid for train with Ground Truth Target
            target will implements Ground Truth target for coco dataset object detection
            in loop, each Hourglass will update weights independently
        '''
        # Normalize image values
        nxt = inputs / 255.0
        nxt = self.preSequence(inputs)

        # calc gradients each Module
        for idx, hourglass in enumerate(self.hourglasses):
            with tf.GradientTape() as tape:
                outputs = hourglass(nxt)
                nxt, mid = outputs
                loss = self.compiled_loss(y_pred=mid, y_true=target)
                
            trainable_vars = hourglass.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(target, mid)

        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs, training=None, mask=None):
        x = inputs
        # Normalize image values

        x = x / 255.0 
        x = self.preSequence(x)
        mid = None
        for hourglass in self.hourglasses:
            x, mid = hourglass(x)
       
        
        max_kps = tf.reduce_max(mid, axis=(inputs.shape[-3], inputs.shape[-2]), keepdims=True)

        mid = tf.cast(tf.where(mid == max_kps, 1, 0), dtype=tf.float32)

        return mid
    
        

        
