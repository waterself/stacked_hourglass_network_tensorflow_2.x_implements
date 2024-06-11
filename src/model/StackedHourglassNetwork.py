from ..layers import layers
from tensorflow import keras
import tensorflow as tf


# class StackedHourglassNet(keras.models.Model):
#     def __init__(self, classes, depth=4, features=256, stacks=8, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stacks = stacks
#         self.PreSequence = layers.PreSequence(features)
#         self.hourglasses = [layers.HourglassWithSuperVision(classes=classes, features=features, depth=depth) for idx in range(stacks)]
#         tf.print('hg_stacks:', len(self.hourglasses) )
    

#     def train_step(self, data):
#         inputs, target = data
#         x = inputs / 255
#         #losses = 0.0
#         with tf.GradientTape(persistent=True) as tape:
#             x = self.PreSequence(x)
#             for idx, hourglass in enumerate(self.hourglasses):
#                 prev = x
#                 if idx == 0:
#                     x, heatmap = hourglass(x)
#                     loss = self.compiled_loss(target, heatmap)

#                     presequence_grads = tape.gradient(loss, self.PreSequence.trainable_variables)
#                     hourglass_grads = tape.gradient(loss, hourglass.trainable_variables)

#                     self.optimizer.apply_gradients(zip(presequence_grads, self.PreSequence.trainable_variables))
#                     self.optimizer.apply_gradients(zip(hourglass_grads, hourglass.trainable_variables))
                
#                 else:
#                     x, heatmap = hourglass(x)
#                     loss = self.compiled_loss(target, heatmap)

#                     #trainable_vars = self.nexts[idx-1].trainable_variables + hourglass.trainable_variables + self.intermediates[idx-1].trainable_variables
#                     next_grads = tape.gradient(loss, self.hourglasses[idx-1].SuperVision.next2.trainable_variables)
#                     mid_grads = tape.gradient(loss, self.hourglasses[idx-1].SuperVision.middle2.trainable_variables)
#                     hourglass_grads = tape.gradient(loss, hourglass.trainable_variables)

#                     self.optimizer.apply_gradients(zip(next_grads, self.hourglasses[idx-1].SuperVision.next2.trainable_variables))
#                     self.optimizer.apply_gradients(zip(mid_grads, self.hourglasses[idx-1].SuperVision.middle2.trainable_variables))
#                     self.optimizer.apply_gradients(zip(hourglass_grads, hourglass.trainable_variables))

#             # losses = [self.compiled_loss(target, pred) for pred in pred_list]
#             # total_loss = tf.reduce_sum(tf.convert_to_tensor(losses, dtype=tf.float32))
        
#         # trainable_vars = self.trainable_variables
#         # gradients = tape.gradient(losses, trainable_vars)
#         # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(target, heatmap)
#         return {m.name: m.result() for m in self.metrics}


        
#     def call(self, inputs, training=None, mask=None):
#         x = inputs / 255
#         tf.print(1)
#         x = self.PreSequence(x)
#         tf.print(2)
#         for idx, hourglass in enumerate(self.hourglasses):
#             tf.print(idx)
#             x, heatmap = hourglass(x)
                    
#         return heatmap
    

# class StackedHourglassNet(keras.models.Model):
#     def __init__(self, classes, depth=4, features=256, stacks=8, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stacks = stacks
#         self.PreSequence = layers.PreSequence(features)
#         self.hourglasses = [layers.Hourglass(depth, features, classes) for idx in range(stacks)]
#         self.intermediates = [layers.IntermediateBlock(features, classes) for idx in range(stacks-1)]
#         self.nexts = [layers.NextBlock(features, classes) for idx in range(stacks - 1)]
#         self.add1 = keras.layers.Add()
#         tf.print('hg_stacks:', len(self.hourglasses) )
       
#     def train_step(self, data):
#         inputs, target = data
#         x = inputs / 255
#         losses = 0.0

#         for idx, hourglass in enumerate(self.hourglasses):
#             with tf.GradientTape(persistent=True) as tape:
#                 if idx == 0:
#                     x = self.PreSequence(x)
#                     prev = x
#                     x = hourglass(x)
#                     x, heatmap = self.intermediates[idx-1](x)
#                     loss = self.compiled_loss(target, heatmap)

#                 else:
#                     x, mid = self.nexts[idx-1]([x, heatmap])
#                     x = self.add1([x, mid, prev])
#                     prev = x
#                     x = hourglass(x)
#                     x, heatmap = self.intermediates[idx-1](x)
#                     loss = self.compiled_loss(target, heatmap)
#                     #trainable_vars = self.nexts[idx-1].trainable_variables + hourglass.trainable_variables + self.intermediates[idx-1].trainable_variables
          
#             if idx == 0: 
#                 presequence_grads = tape.gradient(loss, self.PreSequence.trainable_variables)
#                 hourglass_grads = tape.gradient(loss, hourglass.trainable_variables)
#                 inter_grads = tape.gradient(loss, self.intermediates[idx-1].trainable_variables)

#                 self.optimizer.apply_gradients(zip(presequence_grads, self.PreSequence.trainable_variables))
#                 self.optimizer.apply_gradients(zip(hourglass_grads, hourglass.trainable_variables))
#                 self.optimizer.apply_gradients(zip(inter_grads, self.intermediates[idx-1].trainable_variables))
#             else:
#                 nexts_grads = tape.gradient(loss, self.nexts[idx-1].trainable_variables)
#                 hourglass_grads = tape.gradient(loss, hourglass.trainable_variables)
#                 inter_grads = tape.gradient(loss, self.intermediates[idx-1].trainable_variables)

#                 self.optimizer.apply_gradients(zip(nexts_grads, self.nexts[idx-1].trainable_variables))
#                 self.optimizer.apply_gradients(zip(hourglass_grads, hourglass.trainable_variables))
#                 self.optimizer.apply_gradients(zip(inter_grads, self.intermediates[idx-1].trainable_variables))
#             # losses = [self.compiled_loss(target, pred) for pred in pred_list]
#             # total_loss = tf.reduce_sum(tf.convert_to_tensor(losses, dtype=tf.float32))
        
#         # trainable_vars = self.trainable_variables
#         # gradients = tape.gradient(losses, trainable_vars)
#         # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         self.compiled_metrics.update_state(target, heatmap)
#         return {m.name: m.result() for m in self.metrics}


        
#     def call(self, inputs, training=None, mask=None):
#         x = inputs / 255
        
#         for idx, hourglass in enumerate(self.hourglasses):
#             if idx == 0:
#                 x = self.PreSequence(x)
#                 prev = x
#                 x = hourglass(x)
#                 x, heatmap = self.intermediates[idx-1](x)
#             else:
#                 x, mid = self.nexts[idx-1]([x, heatmap])
#                 x = self.add1([x, mid, prev])
#                 prev = x
#                 x = hourglass(x)
#                 x, heatmap = self.intermediates[idx-1](x)
                    
#         return heatmap


class StackedHourglassNet(keras.models.Model):
    def __init__(self, classes, depth=4, features=256, stacks=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacks = stacks
        self.PreSequence = layers.PreSequence(features)
        self.hourglasses = [layers.HourglassWithSuperVision(classes=classes, features=features, depth=depth) for idx in range(stacks)]
        tf.print('hg_stacks:', len(self.hourglasses))

    def call(self, inputs, training=None, mask=None):
        x = ((inputs / 255) * 0.5) + 0.75
        # x = inputs / 255
        x = self.PreSequence(x)
        output_list = []
        for hourglass in self.hourglasses:
            x, mid = hourglass(x)
            output_list.append(mid)
        
        return output_list
    
    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        return self(x)[-1]
        
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
    #     with tf.GradientTape() as tape:
    #         nxt = self.PreSequence(inputs)

    #         # calc gradients each Module
    #         for idx, hourglass in enumerate(self.hourglasses):
    #             outputs = hourglass(nxt)
    #             nxt, mid = outputs
    #             loss = self.compiled_loss(y_pred=mid, y_true=target)

    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     self.compiled_metrics.update_state(target, mid)

    #     return {m.name: m.result() for m in self.metrics}


