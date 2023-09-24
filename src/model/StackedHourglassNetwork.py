from ..layers import layers
from tensorflow import keras
import tensorflow as tf


CONFIG = {
    ''
}
class StackedHourglassNet(keras.models.Model):
    def __init__(self, classes, stacks=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stacks = stacks
        self.hourglasses = [layers.HourglassWithSuperVision]
        self.intermediateOutput = []
        self.module = layers.Hourglass(2, classes)

        for idx in range(stacks):
            self.hourglasses.append(layers.HourglassWithSuperVision(classes))
    
    def train_step(self, data):
        super().train_step(data)
        inputs, target = data
        
        #TODO: 각 HOURGLASS 모듈이 가중치를 공유하여 학습하지 않도록
        #TODO: 한 HOURGLASS 모듈은 연결된 SuperVision모듈의 중간산출물로만 학습
        #TODO: Grounds Truth Target은 동일하게 적용
        '''
            nxt is forward features
            mid for train with Ground Truth Target
            target will implements Ground Truth target for coco dataset object detection
            in loop, each Hourglass will update weights independently
        '''
        nxt, mid = inputs
        for hourglass in self.hourglasses:
            with tf.GradientTape() as tape:
                nxt, mid = hourglass(nxt)
                loss = self.compiled_loss(target, mid, regularization_losses=self.losses)
            trainable_vars = hourglass.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.compiled_metrics.update_state(target, mid)

        # # # TODO: 각 모듈에 대한 기울기 계산
        # # gradients = tape.gradient(loss, trainable_vars)
        # # TODO: 각 모듈에 대한 기울기 적용
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(target, nxt)
        return {m.name: m.result() for m in self.metrics}
        
    def call(self, inputs, training=None, mask=None):
        super().call(inputs, training, mask)
        x, mid = inputs

        for hourglass in self.hourglasses:
            x, mid = hourglass(x)
        
        return x, mid
    
        

        