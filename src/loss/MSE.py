from keras.utils import losses_utils
import tensorflow as tf
from tensorflow import keras

#keras.losses.mean_squared_error(y_true=target, y_pred=predict)

# class GaussianMSE(keras.losses.Loss):
#     def __init__(self,sigma=1.0,reduction=losses_utils.ReductionV2.AUTO, name=None):
#         super().__init__(reduction, name)
#         self.sigma = sigma

#     def call(self, y_true, y_pred):
#         squared_error = tf.square(y_true - y_pred)
#         gaussian_mse = tf.exp(-squared_error / (2 * self.sigma**2)) / (self.sigma * tf.sqrt(2 * 3.14159265359))
#         return tf.reduce_mean(gaussian_mse)

class MseChannel(keras.losses.Loss):
    def call(self, y_true, y_pred):
        stacked_mse = tf.stack([tf.reduce_mean(tf.square(y_true[:,:,idx] - y_pred[:,:,idx])) for idx in range(y_true.shape[-1])])
        return stacked_mse

# def mean_square_error(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))
#     #return keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
#     # return tf.reduce_mean(mse)
