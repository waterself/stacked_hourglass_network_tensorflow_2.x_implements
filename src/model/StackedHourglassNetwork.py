from ..layers import layers
from tensorflow import keras
import tensorflow as tf



class StackedHourglassNet(keras.models.Model):
    def __init__(self, classes,features=256, stacks=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacks = stacks
        self.hourglasses = [layers.HourglassWithSuperVision(classes)]
        self.intermediateOutput = []
        self.preSequence = keras.layers.Conv2D(filters=features,kernel_size=1, activation='relu')

        for idx in range(stacks):
            self.hourglasses.append(layers.HourglassWithSuperVision(classes))
            print(self.hourglasses[idx])
        print('Init Done')
    
    def train_step(self, data):
        inputs, target = data
        #print('data_type',type(data))
        #print("inputs:", type(inputs))
        #print("target:", type(target))
        #print(data)
        #TODO: 각 HOURGLASS 모듈이 가중치를 공유하여 학습하지 않도록
        #TODO: 한 HOURGLASS 모듈은 연결된 SuperVision모듈의 중간산출물로만 학습
        #TODO: Grounds Truth Target은 동일하게 적용
        '''
            nxt is forward features
            mid for train with Ground Truth Target
            target will implements Ground Truth target for coco dataset object detection
            in loop, each Hourglass will update weights independently
        '''
        nxt = self.preSequence(inputs)
       
      
        #nxt = inputs
        losses_list = []
        # calc gradients each Module
        
        for idx, hourglass in enumerate(self.hourglasses):
            with tf.GradientTape() as tape:
                tape.watch(data)
                outputs = hourglass(nxt)
                nxt, mid = outputs
                loss = self.compiled_loss(y_pred=mid, y_true=target)
                
            trainable_vars = hourglass.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(target, mid)

        # apply loss each module
        # for jdx, hourglass in enumerate(self.hourglasses):
        #     trainable_vars = hourglass.trainable_variables
        #     loss = losses_list[jdx]
        #     gradients = tape.gradient(loss, trainable_vars)
        #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #     self.compiled_metrics.update_state(target, mid)

        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = x / 255.0 
        x = self.preSequence(x)
        mid = None
        for hourglass in self.hourglasses:
            x, mid = hourglass(x)
       
        #Gaussian feature to peak feature
        max_kps = tf.reduce_max(mid, axis=(1, 2), keepdims=True)

        mid = tf.cast(tf.where(mid == max_kps, 1, 0), dtype=tf.float32)

        return mid
    
        

        
