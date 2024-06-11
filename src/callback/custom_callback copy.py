import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

class PoseEstimationCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, validation_data, batch_size, num_samples=4):
        super().__init__()
        self.log_dir = log_dir
        self.validation_data = validation_data
        self.num_samples = num_samples
        self.batch_size = batch_size
        val_data = self.validation_data.take(num_samples // batch_size)
        # self.x_val = tf.concat([image for image, _ in val_data], axis=0)
        # self.y_val = tf.concat([target for _, target in val_data], axis=0)
        self.x_val = []
        self.y_val = []
        for image, target in val_data:
            self.x_val.append(image)
            self.y_val.append(target)
        
        self.x_val = tf.concat(self.x_val, axis=0)
        self.y_val = tf.concat(self.y_val, axis=0)

    def on_epoch_begin(self, epoch, logs=None):
        #image_heigt, image_width, _ = self.x_val.shape[1:]

        writer = tf.summary.create_file_writer(self.log_dir)
        #tf.print("x_val_shape",self.x_val.shape)
        #tf.print("y_val_shape",self.y_val.shape)
        

        #predict = tf.image.resize(predict, (image_heigt, image_width), antialias=True)
        #tf.print("resized_prad_shape:", predict.shape)
        vis_image_list = []
        with writer.as_default():
            for idx in range(self.batch_size):
                image = self.x_val[idx]
                predict = self.model.predict(tf.expand_dims(image, axis=0))
                #tf.summary.image(f"Data_{idx}", tf.cast(tf.expand_dims(self.x_val[idx],axis=0), dtype=tf.uint8), step=epoch)
                # tf.summary.image(f"Target{idx}", tf.cast(tf.reduce_sum(tf.expand_dims(self.y_val[idx], axis=0), axis=-1, keepdims=True)*255, dtype=tf.uint8), step=epoch)
                # tf.summary.image(f"Heatmap{idx}", tf.cast(tf.reduce_sum(tf.expand_dims(predict[idx], axis=0), axis=-1, keepdims=True)*255, dtype=tf.uint8), step=epoch)
                fig_target = plt.figure()
                plt.imshow(tf.reduce_max(self.y_val[idx], axis=-1, keepdims=True))
                tf.summary.image(f"Target{idx}", plot_to_image(fig_target), step=epoch)
                fig_heatmap = plt.figure()
                plt.imshow(tf.reduce_max(predict[0], axis=-1, keepdims=True))
                tf.summary.image(f"Heatmap{idx}", plot_to_image(fig_heatmap), step=epoch)
                
                fig_heatmap.clf()
                fig_target.clf()
                vis_image = tf.cast(image, dtype=tf.uint8).numpy()
                result = predict[0]
                for jdx in range(result.shape[-1]):
                    one_prad = result[:,:,jdx]
                    max_val = tf.reduce_max(one_prad, keepdims=True)
                    cond = tf.equal(one_prad, max_val)
                    #tf.print("where shape:",tf.where(cond).shape)
                    res = tf.where(cond)[0]
                    res = res.numpy()
                    y, x = res[0], res[1]
                    vis_image = cv2.circle(vis_image, (x * 4,y * 4), 3,(0,255,0), cv2.LINE_4) 
                
                vis_image_list.append(tf.cast(tf.expand_dims(vis_image,axis=0)))
                tf.summary.image(f"Predict_{idx}", tf.cast(tf.expand_dims(vis_image,axis=0), dtype=tf.uint8), step=epoch)

        writer.close()

