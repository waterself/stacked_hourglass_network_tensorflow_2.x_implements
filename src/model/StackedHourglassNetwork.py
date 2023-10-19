from ..layers import layers
from tensorflow import keras
import tensorflow as tf


class StackedHourglassNet(keras.models.Model):
    def __init__(self, classes, depth=4, features=256, stacks=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacks = stacks
        self.PreSequence = layers.PreSequence(features)
        self.hourglasses = [layers.HourglassWithSuperVision(classes) for idx in range(stacks)]
        
        print('hg_stacks:', len(self.hourglasses) )
    

    def train_step(self, data):
        inputs, target = data
        tf.print("ts_shape:", inputs.shape)
        x = inputs / 255
        x = self.PreSequence(x)
        with tf.GradientTape() as tape:
            pred_list = self(x)
            losses = [self.compiled_loss(target, pred) for pred in pred_list]
            total_loss = tf.reduce_sum(losses)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(target, pred_list[-1])

        
    def call(self, inputs, training=None, mask=None):
        a = inputs / 255
        tf.print(inputs.shape)
        x = self.PreSequence(a)
        heatmap_list = []
        for hourglass in self.hourglasses:
            x, mid = hourglass(x)
            heatmap_list.append(mid)
        
        return heatmap_list
    
        

        
    # def train_step(self, data):
    #     inputs, target = data
    #     # TODO: Each Modules DO NOT share weights while learning
    #     # TODO: Each Modules will learn with their SuperVision
    #     # TODO: Each Modules use SAME Grounds Truth Target
    #     '''
    #         nxt is forward features
    #         mid for train with Ground Truth Target
    #         target will implements Ground Truth target for coco dataset object detection
    #         in loop, each Hourglass will update weights independently
    #     '''
    #     nxt = inputs / 255
    #     nxt = self.PreSequence(inputs)

    #     # calc gradients each Module
    #     with tf.GradientTape() as tape:
    #         for idx, hourglass in enumerate(self.hourglasses):
    #             outputs = hourglass(nxt)
    #             nxt, mid = outputs
    #             loss = self.compiled_loss(y_pred=mid, y_true=target)

    #             trainable_vars = hourglass.trainable_variables
    #             gradients = tape.gradient(loss, trainable_vars)
    #             self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     self.compiled_metrics.update_state(target, mid)

    #     return {m.name: m.result() for m in self.metrics}
