import tensorflow as tf
from tensorflow import keras

def MeanSqureError(target, predict):
    return tf.keras.losses.MSE(target, predict)

