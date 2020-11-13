import tensorflow as tf
import numpy as np
import collections

from util import binned


def yolo_loss(ground_truth, predicts, num_bbox=2, lambda_coord=5, lambda_noobj=2, epsilon = 1e-6):
    """

    :param ground_truth: ground truth with shape Batch * grid_y * grid_x * (5 + num_classes)
    :param predicts: predicts with shape Batch * grid_y * grid_x * (num_bbox * 5 + num_classes)
    :param lambda_coord: weight of coordinate loss
    :param lambda_noobj: weight of no object confidence loss
    :return: yolo_loss, scalar
    """
    anchor_predicts, p_c = tf.split(predicts, num_or_size_splits=[num_bbox*5, 3], axis=-1)
    anchor_predicts = tf.split(anchor_predicts, num_or_size_splits=num_bbox, axis=-1)
    anchor_predicts = tf.transpose(tf.concat([tf.expand_dims(item, axis=0) for item in anchor_predicts], axis=0),
                                   perm=[1, 2, 3, 0, 4])

    pred_confidence, pred_xy, pred_wh = tf.split(anchor_predicts, num_or_size_splits=[1, 2, 2], axis=-1)
    pred_confidence = tf.squeeze(pred_confidence)

    object_mask = tf.gather(ground_truth, indices=0, axis=3, batch_dims=0)

    ground_truth_expand = tf.expand_dims(ground_truth, axis=3)

    gt_confidence, gt_xy, gt_wh, gt_pc = tf.split(ground_truth_expand, num_or_size_splits=[1, 2, 2, 3], axis=-1)
    gt_confidence = tf.squeeze(gt_confidence)
    gt_pc = tf.squeeze(gt_pc)

    iou_scores = iou_tensor(gt_xy, gt_wh, pred_xy, pred_wh)

    iou_scores_best = tf.reduce_max(iou_scores, axis=-1, keepdims=True)

    iou_mask = tf.cast(tf.logical_and(tf.greater(iou_scores, 0), tf.greater_equal(iou_scores, iou_scores_best)),
                       tf.float32)

    loss_coord_xy = tf.reduce_sum(tf.multiply(tf.expand_dims(iou_mask, axis=-1), tf.subtract(pred_xy, gt_xy) ** 2))
    loss_coord_wh = tf.reduce_sum(tf.multiply(tf.expand_dims(iou_mask, axis=-1),
                                              tf.subtract((pred_wh + epsilon) ** 0.5, (gt_wh + epsilon) ** 0.5) ** 2))
    loss_confidence = tf.add(tf.reduce_sum(tf.multiply(iou_mask,
                                                       tf.subtract(pred_confidence,
                                                                   tf.expand_dims(gt_confidence, axis=-1)) ** 2)),
                             lambda_noobj * tf.reduce_sum(tf.multiply(1 - iou_mask,
                                                                      tf.subtract(pred_confidence,
                                                                                  tf.expand_dims(gt_confidence,
                                                                                                 axis=-1)) ** 2)))

    loss_classes = tf.reduce_sum(tf.multiply(tf.expand_dims(object_mask, axis=-1), tf.subtract(p_c, gt_pc)**2))

    loss = tf.add_n([lambda_coord * loss_coord_xy, lambda_coord * loss_coord_wh, loss_confidence, loss_classes])

    # tf.clip_by_value(loss, clip_value_min=0.0001, clip_value_max=1e6)

    return loss


def iou_tensor(box1_xy, box1_wh, box2_xy, box2_wh, epsilon=1e-4):
    """

    :param box1_xy: with shape Batch * grid_y * grid_x * 1 * 2
    :param box1_wh: with shape Batch * grid_y * grid_x * 1 * 2
    :param box2_xy: with shape Batch * grid_y * grid_x * num_bbox * 2
    :param box2_wh: with shape Batch * grid_y * grid_x * num_bbox * 2
    :param epsilon: epsilon to avoid divided by zero
    :return: iou_scores: with shape Batch * grid_y * grid_x * num_bbox
    """
    image_resolution = tf.constant([1920, 1200], dtype=tf.float32)
    image_resolution = tf.expand_dims(image_resolution, axis=0)
    image_resolution = tf.expand_dims(image_resolution, axis=0)
    image_resolution = tf.expand_dims(image_resolution, axis=0)

    box1_xy_min = tf.multiply(tf.subtract(box1_xy, box1_wh / 2.), image_resolution)
    box1_xy_max = tf.multiply(tf.add(box1_xy, box1_wh / 2.), image_resolution)

    box2_xy_min = tf.multiply(tf.subtract(box2_xy, box2_wh / 2.), image_resolution)
    box2_xy_max = tf.multiply(tf.add(box2_xy, box2_wh / 2.), image_resolution)

    intersection_xy_min = tf.maximum(box1_xy_min, box2_xy_min)
    intersection_xy_max = tf.minimum(box1_xy_max, box2_xy_max)

    box1_x_delta, box1_y_delta = tf.split(tf.subtract(box1_xy_max, box1_xy_min), num_or_size_splits=2, axis=-1)
    box1_area = tf.multiply(box1_x_delta, box1_y_delta)

    box2_x_delta, box2_y_delta = tf.split(tf.subtract(box2_xy_max, box2_xy_min), num_or_size_splits=2, axis=-1)
    box2_area = tf.multiply(box2_x_delta, box2_y_delta)

    xy_delta = tf.subtract(intersection_xy_max, intersection_xy_min)
    xy_delta_mask = tf.cast(tf.greater(xy_delta, 0), tf.float32)

    xy_delta = tf.multiply(xy_delta, xy_delta_mask)
    x_delta, y_delta = tf.split(xy_delta, num_or_size_splits=2, axis=-1)
    intersection_area = tf.multiply(x_delta, y_delta)

    iou_scores = tf.divide(intersection_area,
                           tf.subtract(tf.add(box1_area, box2_area), intersection_area) + epsilon)

    return tf.squeeze(iou_scores)


def mean_avg_pr_tf(ground_truth, predicts, num_bbox=2, iou_threshold=0.5):
    """

    :param ground_truth: ground truth with shape Batch * grid_y * grid_x * (5 + num_classes)
    :param predicts: predicts with shape Batch * grid_y * grid_x * (num_bbox * 5 + num_classes)
    :param num_bbox:
    :param iou_threshold: threshold of iou
    :return: scalar, mean average precision (mAP)
    """
    anchor_predicts, p_c = tf.split(predicts, num_or_size_splits=[num_bbox*5, 3], axis=-1)
    anchor_predicts = tf.split(anchor_predicts, num_or_size_splits=num_bbox, axis=-1)
    anchor_predicts = tf.transpose(tf.concat([tf.expand_dims(item, axis=0) for item in anchor_predicts], axis=0),
                                   perm=[1, 2, 3, 0, 4])

    pred_confidence, pred_xy, pred_wh = tf.split(anchor_predicts, num_or_size_splits=[1, 2, 2], axis=-1)
    pred_confidence = tf.squeeze(pred_confidence)

    object_mask = tf.gather(ground_truth, indices=0, axis=3, batch_dims=0)

    ground_truth_expand = tf.expand_dims(ground_truth, axis=3)

    gt_confidence, gt_xy, gt_wh, gt_pc = tf.split(ground_truth_expand, num_or_size_splits=[1, 2, 2, 3], axis=-1)
    gt_confidence = tf.squeeze(gt_confidence)
    gt_pc = tf.squeeze(gt_pc)

    iou_scores = iou_tensor(gt_xy, gt_wh, pred_xy, pred_wh)

    iou_mask = tf.cast(tf.greater_equal(iou_scores, iou_threshold), tf.float32)     # Batch * grid_y * grid_x * num_bbox


def mean_avg_pr(ground_truth, predicts, fnames, num_classes=3, num_bbox=2, ckpt=100, epsilon=1e-6):
    """

    :param ground_truth: ground truth with shape Batch * grid_y * grid_x * (5 + num_classes)
    :param predicts: predicts with shape Batch * grid_y * grid_x * (num_bbox * 5 + num_classes)
    :param fnames:
    :param num_classes: total of classes
    :param num_bbox:
    :param ckpt:
    :param epsilon:
    :return: scalar, mean average precision (mAP)
            list, list of mAP for different classes
    """
    batch_size, grid_y, grid_x, channel = ground_truth.shape

    anchor_predicts, p_c = np.split(predicts, indices_or_sections=[num_bbox*5], axis=-1)
    anchor_predicts = np.stack(np.split(anchor_predicts, indices_or_sections=num_bbox, axis=-1), axis=3)

    pred_confidence, pred_xy, pred_wh = np.split(anchor_predicts, indices_or_sections=[1, 3], axis=-1)
    pred_confidence = np.squeeze(pred_confidence)   # Batch * grid_y * grid_x * num_bbox

    # p_c with shape Batch * grid_y * grid_x * num_classes
    temp = []
    for i in range(num_classes):
        multi = pred_confidence * np.expand_dims(p_c[:, :, :, i], axis=-1)
        temp.append(np.expand_dims(multi, axis=-1))

    class_confs = np.concatenate(temp, axis=-1)     # Batch * gird_y * grid_x * num_box * num_classes

    gt_confidence, gt_xy, gt_wh, gt_pc = np.split(ground_truth, indices_or_sections=[1, 3, 5], axis=-1)
    gt_confidence = np.squeeze(gt_confidence)   # Batch * grid_y * grid_x
    gt_pc = np.squeeze(gt_pc)   # Batch * grid_y * grid_x * num_classes

    class_map = []
    binned_finals = []
    for c in range(num_classes):
        rows = []
        for k in range(batch_size):
            for i in range(grid_y):
                for j in range(grid_x):
                    for b in range(num_bbox):
                        row = [fnames[k], class_confs[k][i][j][b][c], gt_pc[k][i][j][c]]
                        rows.append(row)

        # save_to_file(rows, filename='val_{}_class_{}.csv'.format(str(ckpt), str(c)))
        scores = [item[1] for item in rows]
        idxs = np.argsort(scores)[::-1]
        rows_sorted = [rows[idx] for idx in idxs]

        gt = [item[2] for item in rows_sorted]
        tp, fp, fn = 0, 0, int(np.sum(gt))   # threshold = 1
        rr = tp / (tp + fn + epsilon)
        pr = tp / (tp + fp + epsilon)
        res = [[rr, pr]]
        for item in rows_sorted:
            if item[2] > 0:
                tp += 1
                fn -= 1
                rr = tp / (tp + fn + epsilon)
            else:
                fp += 1

            pr = tp / (tp + fp + epsilon)
            res.append([rr, pr])

        rr_, pr = res[0]
        final = [[rr_, pr]]
        idx_ = 0
        for i in range(1, len(res)):
            if res[i][0] == rr_:
                final[idx_][1] = max(res[i][1], final[idx_][1])
            else:
                rr_ = res[i][0]
                final.append(res[i])
                idx_ += 1

        length = len(final)
        pr_mx = final[-1][1]
        i = length - 1
        while i >= 0:
            if pr_mx < final[i][1]:
                pr_mx = final[i][1]
            else:
                final[i][1] = pr_mx
            i -= 1

        final_binned = binned(final, column=0, start=0, end=1., bins=20)
        binned_finals.append(final_binned)
        class_map.append(np.mean([item[1] for item in final]))

    return np.mean(class_map), class_map, binned_finals
