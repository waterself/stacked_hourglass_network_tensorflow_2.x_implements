import tensorflow as tf
from tensorflow import keras

# class SelectedMetrics(keras.metrics.Metric):
#     def __init__(self, threshold=0.5, name=None, dtype=None, **kwargs):
#         super().__init__(name, dtype, **kwargs)
#         self.threshold = threshold

#     def call(self, y_true, y_pred, *args, **kwargs):
#         selected = tf.math.count_nonzero(y_true * y_pred)
#         target = tf.math.count_nonzero(y_true)
#         return selected / target

def selected_metrics(y_true, y_pred):
    selected = tf.reduce_sum(tf.cast(tf.math.greater(y_true * y_pred, 0), dtype=tf.int16))
    target = tf.reduce_sum(tf.cast(tf.math.greater(y_true, 0), dtype=tf.int16))
    return selected / target

