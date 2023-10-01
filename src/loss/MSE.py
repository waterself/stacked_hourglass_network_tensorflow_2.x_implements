import tensorflow as tf
from tensorflow import keras

#keras.losses.mean_squared_error(y_true=target, y_pred=predict)

# class mean_square_error(keras.losses.Loss):
#     def call(self, y_true, y_pred):
#         mse = keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
#         return tf.reduce_mean(mse)

def mean_square_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
    #return keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
    # return tf.reduce_mean(mse)

