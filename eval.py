from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras
from src.metric import PCK as pck
from src.loss.heatmap_loss import HeatmapLoss

# load model

model = keras.models.load_model('./8stcks_300epch/', compile=False)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=HeatmapLoss(),
    metrics=[pck, 'acc'],
    )

# load validation dataset
from src.dataset.MPII.read_tfrecord import read_record

val_data = read_record(file_path='./src/dataset/MPII/val_64.tfrecord')
val_data = val_data.shuffle(400)

# collect each point from PCKh
# define dict for collect each joints
# collect each point from PCKh
# define dict for collect each joints
joint_correct_dict = [
    ["r_ankle", [0,0],],
    ["r_knee" , [0,0],],
    ["r_hip", [0,0],],
    ["l_heap", [0,0],],
    ["l_knee", [0,0],],
    ["l_ankle", [0,0],],
    ["pelvis", [0,0],],
    ["thorax", [0,0],],
    ["upper_neck", [0,0],],
    ["head_top", [0,0],],
    ["r_wrist", [0,0],],
    ["r_elbow", [0,0],],
    ["r_shoulder", [0,0],],
    ["l_shoulder", [0,0],],
    ["l_elbow", [0,0],],
    ["l_wrist", [0,0],],
]

# for joint in joint_correct_dict:
#     print(joint)

# test loop for validation data

if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("GPU is available. Using GPU for TensorFlow operations.")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    with tf.device('/GPU:0'):

# test loop for validation data

        area = 100*0.5

        for idx ,(data, target) in enumerate(val_data):
            expanded_data = tf.expand_dims(data, axis=0)
            pred = model.predict(expanded_data, verbose=0)[-1]

            # each channels
            for jdx in range(target.shape[-1]):
                target_joint = target[:,:,jdx]

                # check valid target
                if tf.reduce_all(tf.equal(target_joint, 0)):
                    continue

                pred_joint = tf.squeeze(pred[:,:,jdx], axis=0)

                pred_max_val = tf.reduce_max(pred_joint, keepdims=True)
                pred_cond = tf.equal(pred_joint, pred_max_val)

                target_max_val = tf.reduce_max(target_joint, keepdims=True)
                target_cond = tf.equal(target_joint, target_max_val)


                pred_res = tf.where(pred_cond)[0]
                pred_res = tf.cast(pred_res, dtype=tf.float32)

                true_res = tf.where(target_cond)[0]
                true_res = tf.cast(true_res, dtype=tf.float32)

                distance = tf.norm(true_res - pred_res)
                
                joint_correct_dict[jdx][1][1] += 1
                if distance < area:
                    joint_correct_dict[jdx][1][0] += 1
            print(f"process: {idx}", end='\r', flush=True)
        print("")

corrects = 0
for joint in joint_correct_dict:
    correct = ((joint[1][0] / joint[1][1])*100)
    print(f"{joint[0]}: {correct}, total target :{joint[1][1]}")
    corrects += correct

print(f"total: {correct / len(joint_correct_dict)}")
