﻿import tensorflow as tf
import numpy as np
import cv2
import os
import json

# Define the parameters
num_keypoints = 16  # MPII has 16 joints
variance = 1.0  # You can adjust this based on the scale of your dataset
target_width, target_height = 256, 256 # with paper - training details

# Function to generate a single Gaussian heatmap for a keypoint
def generate_heatmap(keypoint_x, keypoint_y, image_width, image_height, peak_value = 1.0, variance = 1.0):

    target_x = keypoint_x // target_width
    target_y = keypoint_y // target_height

    x = np.arange(0, target_width, 1)
    y = np.arange(0, target_height, 1)
    xv, yv = np.meshgrid(x, y)

    # Calculate the squared distance from each pixel to the keypoint
    distance_squared = (xv - target_x)**2 + (yv - target_y)**2

    # Create a heatmap with zeros
    heatmap = np.zeros((image_height, image_width))

    # Set the value at the keypoint to the peak_value
    heatmap[int(target_y), int(target_x)] = peak_value

    # Optionally, apply Gaussian smoothing to the peak
    heatmap = peak_value * np.exp(-distance_squared / (2.0 * variance ** 2))

    return heatmap

# Load JSON data
#with open('../MPII/annotation/mpii_human_pose_v1_u12_1.json', 'r') as json_file:
with open('./mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.json', 'r') as json_file:
    data = json.load(json_file)

# Define the output directory for TFRecord files
output_directory = './tfrecord_shards'
os.makedirs(output_directory, exist_ok=True)

# Define the maximum number of records per shard (you can adjust this)
records_per_shard = 100

# Create and write TFRecord shards
shard_index = 0
record_index = 0

for annotation in data:
    if record_index % records_per_shard == 0:
        # Create a new TFRecord file for the shard
        shard_filename = os.path.join(output_directory, f'data_shard{shard_index}.tfrecord')
        writer = tf.io.TFRecordWriter(shard_filename)
        shard_index += 1

    # Load and preprocess the image
    img_path = '../MPII/images/' + annotation['img_paths']
    try:
        image = tf.io.read_file(img_path)
    except:
        continue
   
    image = tf.image.decode_image(image, channels=3)


    image_height, image_width, image_channel = image.shape
    # TODO: 이미지 왜곡 해결
    image = tf.image.resize_with_pad(image, target_height, target_width)
    #image = image / 255.0


    # Initialize an empty array to store the heatmaps for all keypoints
    heatmaps = np.zeros((num_keypoints, target_height, target_width))

    # Extract keypoints from the annotation
    keypoints = annotation['joint_self']

    # Iterate through the keypoints and generate a heatmap for each one
    for i in range(num_keypoints):
        keypoint_x, keypoint_y, visibility = keypoints[i]
        if visibility >= 0:  # You can adjust this threshold based on your dataset
            heatmap = generate_heatmap(keypoint_x, keypoint_y, image_width, image_height, variance)
            heatmaps[i] = heatmap

    # Stack the individual heatmaps to create the final ground truth heatmap
    ground_truth_heatmap = np.stack(heatmaps, axis=-1)

    img_path_bytes = bytes(img_path, 'utf-8')
    img_path_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path_bytes]))

    # img_path_feature = tf.train.Features()
    # Encode the image as a bytes feature
    #image_bytes = tf.io.encode_jpeg(image).numpy()
    #image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[image.numpy()]))
    image_feature = tf.io.serialize_tensor(image)
    image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_feature.numpy()]))

    # Encode the ground truth heatmap as a float feature
    #heatmap_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[heatmap_feature]))
    heatmap_feature = tf.io.serialize_tensor(ground_truth_heatmap)
    heatmap_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[heatmap_feature.numpy()]))

    # Create a dictionary with the features
    feature_dict = {
        'img_path' : img_path_feature,
        'image': image_feature,
        'heatmap': heatmap_feature,
    }

    # Create a TF Example
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # Serialize the TF Example and write it to the current shard
    writer.write(example.SerializeToString())
    
    record_index += 1

    # Close the writer and start a new shard if needed
    if record_index % records_per_shard == 0:
        writer.close()

# Close the last shard writer
if record_index % records_per_shard != 0:
    writer.close()

print(f'TFRecord shards have been created in the directory "{output_directory}".')
