﻿import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

class PoseEstimationCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, validation_data, num_samples=4):
        super().__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.num_samples = num_samples
        self.x_val, self.y_val = next(iter(self.validation_data))

    def on_epoch_begin(self, epoch, logs=None):

        writer = tf.summary.create_file_writer(self.log_dir)
        predicted_image = self.model.predict(self.x_val)
        

        with writer.as_default():
            for idx in range(self.num_samples):
                tf.summary.image(f"Data_{idx}", tf.cast(tf.expand_dims(self.x_val[idx],axis=0), dtype=tf.uint8), step=epoch)
                tf.summary.image(f"Target{idx}", tf.cast(tf.reduce_sum(tf.expand_dims(self.y_val[idx], axis=0), axis=-1, keepdims=True)*255, dtype=tf.uint8), step=epoch)
                
                vis_image = tf.cast(self.x_val[idx], dtype=tf.uint8).numpy()
                result_squeezed = predicted_image[idx]
                for jdx in range(result_squeezed.shape[-1]):
                    one_prad = result_squeezed[:,:,jdx]
                    max_val = tf.reduce_max(one_prad, keepdims=True)
                    cond = tf.equal(one_prad, max_val)
                    res = tf.squeeze(tf.where(cond), axis=0)
                    res = res.numpy()
                    y, x = res[0], res[1]
                    vis_image = cv2.circle(vis_image, (x,y), 3,(0,255,0), cv2.LINE_4) 

                tf.summary.image(f"Predict_{idx}", tf.cast(tf.expand_dims(vis_image,axis=0), dtype=tf.uint8), step=epoch)

        writer.close()