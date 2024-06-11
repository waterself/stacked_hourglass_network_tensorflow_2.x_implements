import tensorflow as tf

from tensorflow import keras

import tensorflow_addons as tfa

# class rotate30(keras.layers.Layer):
#     def __init__(self, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
#         super().__init__(trainable, name, dtype, dynamic, **kwargs)
#         self.rotate = keras.layers.RandomRotation(factor=(-0.2, 0.2), fill_mode="nearest", interpolation="nearest")
    
#     def call(self, inputs, *args, **kwargs):
#         x = self.rotate(inputs)
#         return x

@tf.function
def rotate_images_and_heatmaps(images, heatmaps, rotation_range=(-0.3, 0.3)):

    angle = tf.random.uniform([], rotation_range[0], rotation_range[1])

    # rotated_image = tfa.image.rotate(images, angles=angle, interpolation="nearest", fill_mode="nearest")

    # rotated_heatmap = tfa.image.rotate(heatmaps, angles=angle, interpolation="nearest", fill_mode="nearest")

    rotated_image = tfa.image.rotate(images, angles=angle, interpolation="nearest", fill_mode="nearest")

    rotated_heatmap = tfa.image.rotate(heatmaps, angles=angle, interpolation="nearest", fill_mode="nearest")

    return rotated_image, rotated_heatmap