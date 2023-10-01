import tensorflow as tf
import datetime
from tensorflow import keras

from src.model.StackedHourglassNetwork import StackedHourglassNet
from src.layers.layers import HourglassWithSuperVision
from src.loss import MSE


#Load Model

config = {
    'joint_num' : 16,
    'stack_num' : 8,
    'optimizer' : keras.optimizers.Adam(learning_rate=0.000001),
    'loss' : MSE.mean_square_error,
    'epoch' : 100,
    'batch_size': 64,
}

stacked_hourglass = StackedHourglassNet(
    classes=int(config['joint_num']),
    stacks=config['stack_num']
    )

stacked_hourglass.compile(
    optimizer=config['optimizer'],
    loss=config['loss'],
    metrics=['accuracy'],
    )
x = tf.random.uniform(shape=(1,256,256,3), minval=0, maxval=255, dtype=tf.float32)
y = tf.random.uniform(shape=(1,256,256,16), minval=0, maxval=255, dtype=tf.float32)
stacked_hourglass.fit(x, y, epochs=100)
stacked_hourglass.summary()

#hourglass = HourglassWithSuperVision(16)(x)
print("debug")

# from src.dataset.MPII.read_tfrecord import read_record
# train_data = read_record(batch_size=64, file_path='./src/dataset/MPII/train.tfrecord')
# val_data = read_record(batch_size=64, file_path='./src/dataset/MPII/val.tfrecord')

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
#     print("GPU is available. Using GPU for TensorFlow operations.")
    
#     physical_devices = tf.config.list_physical_devices('GPU')
#     if len(physical_devices) > 0:
#         try:
#             tf.config.experimental.set_memory_growth(physical_devices[0], True)
#         except RuntimeError as e:
#             print(e)
#     print("GPU, run Model.fit")
#     with tf.device('/GPU:0'):
#         stacked_hourglass.fit(train_data, epochs=config['epoch'],validation_data=val_data)
#         stacked_hourglass.save('saved_model.h5')
# else:
#     print("No GPU available. Using CPU for TensorFlow operations.")




