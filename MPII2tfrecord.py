import tensorflow as tf
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

    # target_x = keypoint_x // target_width
    # target_y = keypoint_y // target_height

    x = np.arange(0, image_width, 1)
    y = np.arange(0, image_height, 1)
    xv, yv = np.meshgrid(x, y)

    # Calculate the squared distance from each pixel to the keypoint
    distance_squared = (xv - keypoint_x)**2 + (yv - keypoint_y)**2

    # Create a heatmap with zeros
    heatmap = np.zeros((image_height, image_width),dtype=np.float32)

    # Set the value at the keypoint to the peak_value
    heatmap[int(np.clip(keypoint_y, 0, image_height-1)), int(np.clip(keypoint_x, 0, image_width-1))] = peak_value

    # Optionally, apply Gaussian smoothing to the peak
    heatmap = peak_value * np.exp(-distance_squared / (2.0 * variance ** 2))

    return heatmap

# Load JSON data
#with open('./src/dataset/MPII/mpii_human_pose_v1_u12_1.json', 'r') as json_file:
with open('./src/dataset/MPII/mpii_human_pose_v1_u12_1.json', 'r') as json_file:
    data = json.load(json_file)

#open Writer
train_writer = tf.io.TFRecordWriter('./src/dataset/MPII/train_64.tfrecord')
val_writer = tf.io.TFRecordWriter('./src/dataset/MPII/val_64.tfrecord')

total = len(data)
# Each Annotation
for idx, annotation in enumerate(data):
    # Load and preprocess the image
    img_path = '../MPII/images/' + annotation['img_paths']
    try:
        image = tf.io.read_file(img_path)
    except:
        continue
   
    image = tf.image.decode_image(image, channels=3)


    image_height, image_width, image_channel = image.shape
    
    # Initialize an empty array to store the heatmaps for all keypoints, witn original Resolution
    heatmaps = np.zeros((num_keypoints, image_height, image_width),dtype=np.float32)

    # Extract keypoints from the annotation
    keypoints = annotation['joint_self']

    # Iterate through the keypoints and generate a heatmap for each one
    for i in range(num_keypoints):
        keypoint_x, keypoint_y, visibility = keypoints[i]
        if keypoint_x == 0 and keypoint_y == 0:
            continue  
        if visibility >= 0:
            heatmap = generate_heatmap(keypoint_x, keypoint_y, image_width, image_height, variance)
            heatmaps[i] = heatmap


    # Stack the individual heatmaps to create the final ground truth heatmap
    ground_truth_heatmap = np.stack(heatmaps, axis=-1)

    # Resize With pad with Same Size of Data
    image = tf.image.resize(image, (target_height, target_width), antialias=True)
    ground_truth_heatmap = tf.image.resize(ground_truth_heatmap, (64, 64), antialias=True)


    img_path_bytes = bytes(img_path, 'utf-8')
    img_path_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_path_bytes]))

    image_feature = tf.io.serialize_tensor(image)    
    image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_feature.numpy()]))

    # Encode the ground truth heatmap as a float feature
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
    if annotation['is_validation'] == 0.0:
        train_writer.write(example.SerializeToString())
    else:
        val_writer.write(example.SerializeToString())
    print(f"process:{idx}/{total}", end='\r', flush=True)
    
    


train_writer.close()
val_writer.close()

print(f'TFRecord shards have been created in the directory ./src/dataset/MPII/')

