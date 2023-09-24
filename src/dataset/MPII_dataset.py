import json
import tensorflow as tf

target_json_path = "./MPII/mpii_human_pose_v1_u12_1.json"

annotation_json = json.loads(target_json_path)

print(annotation_json)

class MPII_dataset(tf.data.Dataset):
    def _generator(num_samples):
        