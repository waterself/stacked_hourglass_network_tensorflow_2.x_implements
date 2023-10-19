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
    selected = 0
    for batch in range(y_true.shape[0]):
        result_bacth = y_pred[batch]
        for jdx in range(result_bacth.shape[-1]):
            one_prad = result_bacth[:,:,jdx]
            max_val = tf.reduce_max(one_prad, keepdims=True)
            cond = tf.cast(tf.equal(one_prad, max_val), dtype=tf.int16)
            selected += tf.math.count_nonzero(cond * y_true[batch, :,:,jdx])

    return selected / y_true.shape[-1]

