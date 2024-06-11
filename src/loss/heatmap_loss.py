from keras.utils import losses_utils
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

class HeatmapLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        l = K.square(y_pred - y_true)
        l = K.mean(l, axis=[1, 2, 3])
        return l