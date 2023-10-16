import numpy as np
import datetime
import os
import tensorflow as tf
from tensorflow import keras

from src.model.StackedHourglassNetwork import StackedHourglassNet
from src.layers.layers import HourglassWithSuperVision
from src.loss import MSE
from src.callback import custom_callback
from src.metric.selected_metrics import selected_metrics as jsp


#Load Model

config = {
    'joint_num' : 16,
    'stack_num' : 8, 
    'resolution' : 256,
    'optimizer' : keras.optimizers.Adam,
    'loss' : MSE.mean_square_error,
    'epoch' : 10,
    'batch_size': 4, 

}

stacked_hourglass = StackedHourglassNet(
    classes=int(config['joint_num']),
    stacks=config['stack_num']
    )

stacked_hourglass.compile(
    optimizer=config['optimizer'],
    loss=config['loss'],
    metrics=[jsp,'mae'],
    )

from src.dataset.MPII.read_tfrecord import read_record, parse_serial
import matplotlib.pyplot as plt
train_data = read_record(file_path='./src/dataset/MPII/train.tfrecord')
val_data = read_record(file_path='./src/dataset/MPII/val.tfrecord')

train_data = train_data.shuffle(2000)
val_data = val_data.shuffle(400)

train_data = train_data.padded_batch(
    config['batch_size'],
    padded_shapes=((config['resolution'], config['resolution'], 3), (config['resolution'], config['resolution'], 16)))

val_data = val_data.padded_batch(
    config['batch_size'],
    padded_shapes=((config['resolution'], config['resolution'], 3), (config['resolution'], config['resolution'], 16)))

print(type(train_data))
#train_data.repeat(config['epoch'])
#val_data.repeat(config['epoch'])

first_batch = train_data.take(1)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#file_writer = tf.summary.create_file_writer(log_dir)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
val_callback = custom_callback.PoseEstimationCallback(log_dir=log_dir, 
                                                      validation_data=val_data, 
                                                      num_samples=3)
es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                min_delta=0.00001,
                                                patience = 5,
                                                mode='auto')

stacked_hourglass.load_weights(checkpoint_path)


# Iterate through the batch and print the shape of the first sample
for idx, (sample_image, sample_heatmap) in enumerate(first_batch):
    print("Shape of the first sample image:", sample_image.shape)
    print("Shape of the first sample heatmap:", sample_heatmap.shape)


if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("GPU is available. Using GPU for TensorFlow operations.")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    print("GPU, run Model.fit")
    with tf.device('/GPU:0'):
        for sample_image, sample_heatmap in first_batch:
            print(f"image_shape{sample_image.shape}")    
            print(f"heatmap_shape{sample_heatmap.shape}")
            print(np.max(sample_image.numpy()))
            print(np.min(sample_image.numpy()))
            test_out = stacked_hourglass(sample_image)
            print("test_out_shape:", test_out.shape)

        stacked_hourglass.fit(train_data.repeat(config['epoch']), 
                epochs=config['epoch'],
                validation_data=val_data.repeat(config['epoch']), 
                batch_size=config['batch_size'],
                validation_batch_size=config['batch_size'],
                steps_per_epoch=15400 // config['batch_size'],
                validation_steps=3785 // config['batch_size'],
                callbacks = [tensorboard_callback, 
                             cp_callback,
                             val_callback,
                             es_callback],
                )
        stacked_hourglass.save(f"./{config['stack_num']}stcks_{config['epoch']}epch/")
else:
    print("No GPU available. Using CPU for TensorFlow operations.")


