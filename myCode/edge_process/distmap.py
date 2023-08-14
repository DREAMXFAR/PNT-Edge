import numpy as np
import cv2


def gen_shift(label, size):
    # from reanno to orig annotation: mappings ~= reanno pixels
    dist_2, labels = cv2.distanceTransformWithLabels(1 - label, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    # filter outliers
    threshold = 15

    index = np.copy(labels)
    if np.max(labels) == 0:
        return np.zeros_like(label), np.zeros_like(label)

    index[1 - label > 0] = 0
    place = np.argwhere(index > 0)
    nearCord = place[labels - 1, :]
    x = nearCord[:, :, 0]
    y = nearCord[:, :, 1]

    ### mapping function
    grid_y, grid_x = np.meshgrid(range(size), range(size))

    delta_y = (y - grid_y)
    delta_x = (x - grid_x)

    delta_x[dist_2 > threshold] = 0
    delta_y[dist_2 > threshold] = 0

    return -delta_y, -delta_x


def min_dist_mapping(source, target):
    # mapping source pixels to target pixels, source < target
    dist_2, labels = cv2.distanceTransformWithLabels(1 - target, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    # filter outliers
    threshold = 15

    # if labels are all zero, return zero map
    index = np.copy(labels)
    if np.max(labels) <= 0:
        return np.zeros_like(target), np.zeros_like(target), np.zeros_like(target)

    # generate nearest position map
    index[1 - target > 0] = 0
    # index[1 - source > 0] = 0
    place = np.argwhere(index > 0)
    nearCord = place[labels - 1, :]
    x = nearCord[:, :, 0]
    y = nearCord[:, :, 1]

    # set the mapped target as 1 and set other pixels as 0
    mapped_target = np.zeros_like(target)
    source_nonzero_points = np.nonzero(source)
    mapped_target[x[source_nonzero_points], y[source_nonzero_points]] = 1

    # generate shift from scource to target
    # notion: the flow is defined at the pixels of source, so the field accord to source=f(target, flow)
    size = target.shape[0]
    grid_y, grid_x = np.meshgrid(range(size), range(size))
    delta_y = (y - grid_y) * source  # only sample shifts of valid source pixels
    delta_x = (x - grid_x) * source

    delta_x[dist_2 > threshold] = 0
    delta_y[dist_2 > threshold] = 0

    # x: row, y: column, from source to target
    return mapped_target, delta_x, delta_y


def gen_nearest_point_map(label):
    # label in 0, 1, compute dist to pixels=1
    dist, labels = cv2.distanceTransformWithLabels(1 - label, cv2.DIST_L1, 3, labelType=cv2.DIST_LABEL_PIXEL)

    if np.max(labels) <= 0:
        size = label.shape[0]
        grid_y, grid_x = np.meshgrid(range(size), range(size))
        dist = np.zeros_like(label)
        return grid_x, grid_y, dist

    index = np.copy(labels)
    # generate nearest position map
    index[1 - label > 0] = 0
    place = np.argwhere(index > 0)
    nearCord = place[labels - 1, :]
    x = nearCord[:, :, 0]
    y = nearCord[:, :, 1]

    return x, y, dist


def min_dist_mapping_tflow(source, target):
    # filter outliers
    threshold = 15
    # mapping source pixels to target pixels, source < target
    x, y, dist= gen_nearest_point_map(target)

    # set the mapped target as 1 and set other pixels as 0
    mapped_target_mask = np.zeros_like(target)
    source_nonzero_points = np.nonzero(source)
    mapped_target_mask[x[source_nonzero_points], y[source_nonzero_points]] = 1

    # generate shift from target to source, if the flow is defined at the pixels of source, so the field accord to source=f(target, flow)
    x, y, _ = gen_nearest_point_map(source)
    size = target.shape[0]
    grid_y, grid_x = np.meshgrid(range(size), range(size))
    delta_y = (y - grid_y) * mapped_target_mask  # only sample shifts of valid source pixels
    delta_x = (x - grid_x) * mapped_target_mask

    delta_x[dist > threshold] = 0
    delta_y[dist > threshold] = 0
    # x: row, y: column, from source to target
    return mapped_target_mask, delta_x, delta_y


