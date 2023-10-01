import tensorflow as tf
import os

# def get_records_path(path):
#     file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#     return file_names

def _parse_serial(serialized_string):
    feature_description = {
        'img_path': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'heatmap': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_string,
                                         feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    heatmap = tf.io.parse_tensor(example['heatmap'], out_type=tf.float32)
    return image, heatmap


def read_record(batch_size : int, file_path):
    raw_dataset = tf.data.TFRecordDataset(filenames=file_path).map(_parse_serial).batch(batch_size)
    return raw_dataset

