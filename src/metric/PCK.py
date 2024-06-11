import tensorflow as tf

@tf.function
def PCK(y_true, y_pred, threshold = 0.5):
    area = 25*threshold
    isCorrect = 0
    total = 0
    for joint_idx in range(y_pred.shape[-1]):
        pred_heatmap = y_pred[:,:,joint_idx]
        true_heatmap = y_true[:,:,joint_idx]
        # TODO: How to know target is empty?
        if tf.reduce_all(tf.equal(true_heatmap, 0)):
            continue

        pred_max_val = tf.reduce_max(pred_heatmap, keepdims=True)
        pred_cond = tf.equal(pred_heatmap, pred_max_val)

        true_max_val = tf.reduce_max(true_heatmap, keepdims=True)
        true_cond = tf.equal(true_heatmap, true_max_val)


        pred_res = tf.where(pred_cond)[0]
        pred_res = tf.cast(pred_res, dtype=tf.float32)
        #pred_y, pred_x = pred_res[0], pred_res[1]

        true_res = tf.where(true_cond)[0]
        true_res = tf.cast(true_res, dtype=tf.float32)
        #true_y, true_x = true_res[0], true_res[1]

        distance = tf.norm(true_res - pred_res)
        total +=1
        if distance < area:
            isCorrect+=1

    return isCorrect / total