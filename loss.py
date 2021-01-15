import tensorflow as tf
import numpy as np
import collections
import os

from util import binned, sigmoid, soft_max

tf_float = tf.float16


def construct_bbox_params():
    img_width = 1920
    img_height = 1200
    grid_res = 48
    width_cn = img_width // grid_res
    height_cn = img_height // grid_res
    num_bbox = 5

    anchor_priors = [[676, 583], [377, 327], [205, 161], [113, 83], [51, 50]]

    y = np.zeros((height_cn, width_cn, num_bbox, 4))
    y_p = np.zeros((height_cn, width_cn, num_bbox, 4))
    for i in range(height_cn):
        for j in range(width_cn):
            for b in range(num_bbox):
                y[i][j][b] = np.array([j * grid_res, i * grid_res, anchor_priors[b][0], anchor_priors[b][1]])
                y_p[i][j][b] = y[i][j][b]/np.array([img_width, img_height, img_width, img_height])

    return y, y_p


true_size, p_size = construct_bbox_params()
c_xy, p_wh = np.split(true_size, indices_or_sections=2, axis=-1)
c_xy_p, p_wh_p = np.split(p_size, indices_or_sections=2, axis=-1)


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


def yolo_loss_v2(ground_truth, predicts, num_bbox=5, num_classes=3, batch_size=4,
                 lambda_coord_xy=1., lambda_coord_wh=1.,  lambda_obj=5., lambda_noobj=0.,
                 lambda_prior=0., lambda_conf=1., epsilon=1e-4):
    """

    :param ground_truth: ground truth with shape Batch * grid_y * grid_x * [num_bbox * (5 + num_classes)]]
    :param predicts: predicts with shape Batch * grid_y * grid_x * [num_bbox * (5 + num_classes)]
    :param num_bbox:
    :param num_classes:
    :param batch_size:
    :param lambda_coord_xy: weight of coordinate xy loss
    :param lambda_coord_wh: weight of coordinate wh loss
    :param lambda_obj:
    :param lambda_noobj: weight of no object confidence loss
    :param lambda_prior: default to 0.005
    :param lambda_conf:
    :param epsilon:
    :return: yolo_loss_v2, scalar
    """
    # tf.print(tf.reduce_min(predicts), tf.reduce_max(predicts))
    # tf.print(tf.reduce_min(ground_truth), tf.reduce_max(ground_truth))

    anchor_predicts = tf.clip_by_value(predicts, -10, 10)
    anchor_predicts = tf.split(anchor_predicts, num_or_size_splits=num_bbox, axis=-1)
    anchor_predicts = tf.concat([tf.expand_dims(item, axis=3) for item in anchor_predicts], axis=3)
    gts = tf.clip_by_value(ground_truth, -10, 10)
    gts = tf.split(gts, num_or_size_splits=num_bbox, axis=-1)
    gts = tf.concat([tf.expand_dims(item, axis=3) for item in gts], axis=3)

    inds = tf.where(gts[..., 0] > 0.5)

    pred_confidence, pred_xy, pred_wh, p_c = tf.split(anchor_predicts, num_or_size_splits=[1, 2, 2, num_classes], axis=-1)
    pred_confidence = tf.squeeze(pred_confidence)   # Batch * grid_y * grid_x * num_bbox
    pred_confidence = tf.math.sigmoid(pred_confidence)
    p_c = tf.math.softmax(p_c)
    sigma_xy = tf.math.sigmoid(pred_xy)
    pred_wh = tf.clip_by_value(pred_wh, -3, 3)
    # pred_wh = tf.math.sigmoid(pred_wh) * 2 - 1  # 这是什么操作？！

    object_mask = tf.gather(gts, indices=0, axis=-1, batch_dims=0)  # Batch * grid_y * grid_x * num_bbox

    gt_confidence, gt_xy, gt_wh, gt_pc = tf.split(gts, num_or_size_splits=[1, 2, 2, num_classes], axis=-1)
    sigma_xy_gt = tf.math.sigmoid(gt_xy)

    c_xy_t = tf.convert_to_tensor(c_xy, dtype=tf_float)
    p_wh_t = tf.convert_to_tensor(p_wh, dtype=tf_float)
    p_wh_p_t = tf.convert_to_tensor(p_wh_p, dtype=tf_float)
    p_xy_dep, p_wh_dep = deparameterize_tf(pred_xy, pred_wh, c_xy_t, p_wh_t, isbatch=True)
    gt_xy_dep, gt_wh_dep = deparameterize_tf(gt_xy, gt_wh, c_xy_t, p_wh_t, isbatch=True)

    best_ious = best_ious_tf(gt_confidence, gt_xy_dep, gt_wh_dep, p_xy_dep, p_wh_dep, batch_size)
    best_ious = tf.clip_by_value(best_ious, 0.0001, 0.9999)
    # tf.print(tf.reduce_min(best_ious), tf.reduce_max(best_ious))
    # best_ious with shape (batch, grid_y, grid_x, num_bbox)

    object_detections = tf.cast(tf.greater_equal(best_ious, 0.5), dtype=tf_float)
    count = tf.reduce_sum(object_detections)
    obj_count = tf.reduce_sum(object_mask)

    noobj_count = tf.reduce_sum((1. - object_detections) / 100.) * 100
    # tf.print(obj_count, noobj_count, count)
    # tf.print(tf.reduce_min(pred_wh), tf.reduce_max(pred_wh), tf.reduce_min(gt_wh), tf.reduce_max(gt_wh))

    avg_iou = tf.reduce_sum(tf.multiply(best_ious, object_mask)) / (obj_count + epsilon)
    avg_obj = tf.reduce_sum(tf.multiply(pred_confidence, object_mask)) / (obj_count + epsilon)
    avg_anyobj = tf.reduce_sum(pred_confidence) / tf.cast(tf.reduce_prod(tf.shape(ground_truth)[:-1]), dtype=tf_float)
    recall = tf.reduce_sum(tf.cast(tf.multiply(best_ious, object_mask) > 0.5, dtype=tf_float)) / (obj_count + epsilon)
    avg_cat = tf.reduce_sum(p_c * gt_pc) / (obj_count + epsilon)

    loss_coord_xy = lambda_coord_xy * tf.reduce_sum(tf.multiply(tf.expand_dims(object_mask, axis=-1),
                                                                tf.subtract(sigma_xy, sigma_xy_gt) ** 2))
    loss_coord_wh = lambda_coord_wh * tf.reduce_sum(tf.multiply(tf.expand_dims(object_mask, axis=-1),
                                                                tf.subtract(pred_wh, gt_wh) ** 2))
    # picked_sigma_xy = tf.gather_nd(sigma_xy, indices=inds)
    # picked_sigma_xy_gt = tf.gather_nd(sigma_xy_gt, indices=inds)
    # picked_pred_wh = tf.gather_nd(pred_wh, indices=inds)
    # picked_gt_wh = tf.gather_nd(gt_wh, indices=inds)

    # tf.print(tf.reduce_min(pred_wh), tf.reduce_max(pred_wh), tf.reduce_min(sigma_xy), tf.reduce_max(sigma_xy))
    prior_loss_xy = tf.reduce_sum((sigma_xy - 0.5) ** 2)
    prior_loss_wh = tf.reduce_sum((pred_wh/3.) ** 2)

    # temp1 = tf.multiply(object_mask, pred_confidence)
    # temp2 = tf.multiply(object_mask, best_ious)
    # tf.print(tf.reduce_max(temp1), tf.reduce_max(temp2))
    loss_obj = lambda_obj * tf.reduce_sum(tf.multiply(object_mask,
                                                      tf.subtract(pred_confidence, best_ious) ** 2))
    noobj_weights = tf.multiply((1 - object_detections), (1 - object_mask))
    loss_noobj = lambda_noobj * tf.reduce_sum(tf.multiply(noobj_weights, pred_confidence ** 2)) / (noobj_count + epsilon)

    loss_classes = lambda_conf * tf.reduce_sum(tf.multiply(tf.expand_dims(object_mask, axis=-1),
                                                           tf.subtract(gt_pc, p_c) ** 2))
    loss_prior = lambda_prior * tf.add(prior_loss_xy, prior_loss_wh)

    # tf.print(loss_prior, prior_loss_xy, prior_loss_wh)
    tf.print(loss_coord_xy, loss_coord_wh, loss_obj, loss_noobj, loss_classes)
    # tf.print(avg_iou, avg_obj, avg_anyobj, avg_cat, recall)

    loss = tf.add_n([loss_noobj, loss_classes, loss_obj,
                    loss_coord_xy, loss_coord_wh, loss_prior])

    # file_writer = tf.summary.create_file_writer(os.path.join("F:\Jupyter\Kaggle\Projects\ObjectDetection\yolo\model", 'logs'))
    # tf.summary.scalar('loss_prior', data=loss)

    return loss


def best_ious_tf(gt_confidence, gt_xy, gt_wh, pred_xy, pred_wh, batch_size):
    """

    :param gt_confidence: (Batch, grid_x, grid_y, num_bbox, 1)
    :param gt_xy: (Batch, grid_x, grid_y, num_bbox, 2), 0-1, de-parameterized
    :param gt_wh: (Batch, grid_x, grid_y, num_bbox, 2), 0-1, de-parameterized
    :param pred_xy: (Batch, grid_x, grid_y, num_bbox, 2), 0-1, de-parameterized
    :param pred_wh: (Batch, grid_x, grid_y, num_bbox, 2), 0-1, de-parameterized
    :param batch_size:
    :return: best ious, tensor, (Batch, grid_x, grid_y, num_bbox)
    """
    best_ious = []
    for i in range(batch_size):
        ind = tf.where(gt_confidence[i][..., 0] > 0.5)
        true_xy = tf.gather_nd(gt_xy[i], indices=ind)
        true_xy = tf.reshape(true_xy, [1, 1, 1, -1, 2])
        true_wh = tf.gather_nd(gt_wh[i], indices=ind)
        true_wh = tf.reshape(true_wh, [1, 1, 1, -1, 2])

        ious = iou_tensor(tf.expand_dims(pred_xy[i], axis=3), tf.expand_dims(pred_wh[i], axis=3),
                          true_xy, true_wh, multi=100)
        ious = tf.clip_by_value(ious, 0.0001, 0.9999)
        best_iou = tf.reduce_max(ious, axis=3, keepdims=False)
        # best_iou with shape (?, ?, ?)
        best_ious.append(best_iou)

    return tf.stack(best_ious, axis=0)  # with shape (batch, ?, ?, ?)


def class_maps(batch_size, num_classes, grid_y, grid_x, num_bbox, fnames, class_confs, gt_pc, epsilon=1e-6):
    """

    :param batch_size:
    :param num_classes:
    :param grid_y:
    :param grid_x:
    :param num_bbox:
    :param fnames:
    :param class_confs:
    :param gt_pc:
    :param epsilon:
    :return:
    """
    class_map = []
    binned_finals = []
    for c in range(num_classes):
        rows = []
        for k in range(batch_size):
            for i in range(grid_y):
                for j in range(grid_x):
                    for b in range(num_bbox):
                        row = [fnames[k], class_confs[k][i][j][b][c], gt_pc[k][i][j][b][c]]
                        rows.append(row)

        # save_to_file(rows, filename='val_{}_class_{}.csv'.format(str(ckpt), str(c)))
        scores = [item[1] for item in rows]
        idxs = np.argsort(scores)[::-1]
        rows_sorted = [rows[idx] for idx in idxs]

        gt = [item[2] for item in rows_sorted]
        tp, fp, fn = 0, 0, int(np.sum(gt))  # threshold = 1
        rr = tp / (tp + fn + epsilon)
        pr = tp / (tp + fp + epsilon)
        res = [[rr, pr]]
        for item in rows_sorted:
            if item[2] == 1:
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

    return class_map, binned_finals


def label_decode_tf(xy, wh, image_resolution):
    """
    # TODO: not used
    :param xy: tensor, 0-1
    :param wh: tensor, 0-1
    :param image_resolution: const tensor, image width and height
    :return:
    """
    xy_min = tf.multiply(tf.subtract(xy, wh / 2.), image_resolution)
    xy_min = tf.clip_by_value(xy_min, 0, 1)
    xy_max = tf.multiply(tf.add(xy, wh / 2.), image_resolution)
    xy_max = tf.clip_by_value(xy_max, 0, 1)

    return xy_min, xy_max


def iou_tensor(box1_xy, box1_wh, box2_xy, box2_wh, epsilon=1e-4, multi=1):
    """

    :param box1_xy: with shape Batch * grid_y * grid_x * 1 * 2
    :param box1_wh: with shape Batch * grid_y * grid_x * 1 * 2
    :param box2_xy: with shape Batch * grid_y * grid_x * num_bbox * 2
    :param box2_wh: with shape Batch * grid_y * grid_x * num_bbox * 2
    :param epsilon: epsilon to avoid divided by zero
    :return: iou_scores: with shape Batch * grid_y * grid_x * num_bbox
    """
    box1_xy_min = tf.clip_by_value(box1_xy*multi - box1_wh*multi * 0.5, 0, multi)
    box1_xy_max = tf.clip_by_value(box1_xy*multi + box1_wh*multi * 0.5, 0, multi)
    box2_xy_min = tf.clip_by_value(box2_xy*multi - box2_wh*multi * 0.5, 0, multi)
    box2_xy_max = tf.clip_by_value(box2_xy*multi + box2_wh*multi * 0.5, 0, multi)

    intersect_xy_min = tf.maximum(box1_xy_min, box2_xy_min)
    intersect_xy_max = tf.minimum(box1_xy_max, box2_xy_max)

    box1_w, box1_h = tf.split(box1_wh*multi, num_or_size_splits=2, axis=-1)
    box1_area = tf.multiply(box1_w, box1_h)

    box2_w, box2_h = tf.split(box2_wh*multi, num_or_size_splits=2, axis=-1)
    box2_area = tf.multiply(box2_w, box2_h)

    xy_delta = tf.subtract(intersect_xy_max, intersect_xy_min)
    x_delta, y_delta = tf.split(xy_delta, num_or_size_splits=2, axis=-1)
    xy_delta_mask = tf.cast(tf.logical_and(tf.greater_equal(x_delta, 0), tf.greater_equal(y_delta, 0)), dtype=tf_float)
    intersect_area = tf.multiply(x_delta, y_delta)
    intersect_area = tf.multiply(xy_delta_mask, intersect_area)

    iou_scores = tf.divide(intersect_area,
                           tf.subtract(tf.add(box1_area, box2_area), intersect_area) + epsilon)

    return tf.squeeze(iou_scores, axis=-1)


def deparameterize_tf(t_xy, t_wh, center_xy, power_wh, grid_resolution=48, isbatch=True):
    """

    :param t_xy:
    :param t_wh:
    :param center_xy:
    :param power_wh:
    :param grid_resolution:
    :param isbatch:
    :return: xy, wh, 0-1, normed
    """
    if isbatch:
        c_xy_ext = tf.expand_dims(center_xy, axis=0)
        p_wh_ext = tf.expand_dims(power_wh, axis=0)
    else:
        c_xy_ext = center_xy
        p_wh_ext = power_wh

    wh_const = tf.constant([1920, 1200], dtype=tf_float)
    wh_const = tf.expand_dims(wh_const, axis=0)
    wh_const = tf.expand_dims(wh_const, axis=0)
    if isbatch:
        wh_const = tf.expand_dims(wh_const, axis=0)

    xy = tf.divide(tf.add(tf.sigmoid(t_xy) * grid_resolution, c_xy_ext),  wh_const)
    wh = tf.divide(tf.multiply(tf.math.exp(t_wh), p_wh_ext), wh_const)
    wh = tf.clip_by_value(wh, 0.0001, 0.9999)

    return xy, wh


# def mean_avg_pr_tf(ground_truth, predicts, num_bbox=2, iou_threshold=0.5):
#     """
#
#     :param ground_truth: ground truth with shape Batch * grid_y * grid_x * (5 + num_classes)
#     :param predicts: predicts with shape Batch * grid_y * grid_x * (num_bbox * 5 + num_classes)
#     :param num_bbox:
#     :param iou_threshold: threshold of iou
#     :return: scalar, mean average precision (mAP)
#     """
#     anchor_predicts, p_c = tf.split(predicts, num_or_size_splits=[num_bbox*5, 3], axis=-1)
#     anchor_predicts = tf.split(anchor_predicts, num_or_size_splits=num_bbox, axis=-1)
#     anchor_predicts = tf.transpose(tf.concat([tf.expand_dims(item, axis=0) for item in anchor_predicts], axis=0),
#                                    perm=[1, 2, 3, 0, 4])
#
#     pred_confidence, pred_xy, pred_wh = tf.split(anchor_predicts, num_or_size_splits=[1, 2, 2], axis=-1)
#     pred_confidence = tf.squeeze(pred_confidence)
#
#     object_mask = tf.gather(ground_truth, indices=0, axis=3, batch_dims=0)
#
#     ground_truth_expand = tf.expand_dims(ground_truth, axis=3)
#
#     gt_confidence, gt_xy, gt_wh, gt_pc = tf.split(ground_truth_expand, num_or_size_splits=[1, 2, 2, 3], axis=-1)
#     gt_confidence = tf.squeeze(gt_confidence)
#     gt_pc = tf.squeeze(gt_pc)
#
#     iou_scores = iou_tensor(gt_xy, gt_wh, pred_xy, pred_wh)
#
#     iou_mask = tf.cast(tf.greater_equal(iou_scores, iou_threshold), tf.float32)
# Batch * grid_y * grid_x * num_bbox


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
    gt_pc = np.squeeze(gt_pc)   # Batch * grid_y * grid_x * num_classes

    class_map, binned_finals = class_maps(batch_size, num_classes, grid_y, grid_x, num_bbox, fnames, class_confs, gt_pc)

    return np.mean(class_map), class_map, binned_finals


def mean_avg_pr_v2(ground_truth, predicts, fnames, num_classes=3, num_bbox=2, ckpt=100, epsilon=1e-6):
    """

    :param ground_truth: ground truth with shape Batch * grid_y * grid_x * (num_bbox * (5 + num_classes))
    :param predicts: predicts with shape Batch * grid_y * grid_x * (num_bbox * (5 + num_classes))
    :param fnames:
    :param num_classes: total of classes
    :param num_bbox:
    :param ckpt:
    :param epsilon:
    :return: scalar, mean average precision (mAP)
            list, list of mAP for different classes
    """
    batch_size, grid_y, grid_x, channel = ground_truth.shape

    anchor_predicts = np.split(predicts, indices_or_sections=num_bbox, axis=-1)
    anchor_predicts = np.stack(anchor_predicts, axis=3)

    pred_confidence, pred_xy, pred_wh, p_c = np.split(anchor_predicts, indices_or_sections=[1, 3, 5], axis=-1)
    pred_confidence = np.squeeze(sigmoid(pred_confidence))   # Batch * grid_y * grid_x * num_bbox

    # p_c and class_confs with shape Batch * grid_y * grid_x * num_bbox * num_classes

    p_c = soft_max(p_c, axis=-1)
    class_confs = p_c * np.expand_dims(pred_confidence, axis=-1)

    gts = np.split(ground_truth, indices_or_sections=num_bbox, axis=-1)
    gts = np.stack(gts, axis=3)
    gt_confidence, gt_xy, gt_wh, gt_pc, noobj_mask = np.split(gts,
                                                              indices_or_sections=[1, 3, 5, 5+num_classes], axis=-1)

    class_map, binned_finals = class_maps(batch_size, num_classes, grid_y, grid_x, num_bbox, fnames, class_confs, gt_pc)

    return np.mean(class_map), class_map, binned_finals
