import numpy as np
import torch


def rbbox2circumhbbox(rbboxes):
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w - sina * h
    hbbox_h = - sina * w + cosa * h
    # -pi/2 < a <= 0, so cos(a)>0, sin(a)<0
    hbboxes = rbboxes.clone().detach()
    hbboxes[:, 2::5] = hbbox_h
    hbboxes[:, 3::5] = hbbox_w
    hbboxes[:, 4::5] = -np.pi / 2
    return hbboxes


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 6)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox_flip(bboxes,
               img_shape,
               direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal" and
            "vertical". Default: "horizontal"


    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 5 == 0
    orig_shape = bboxes.shape
    bboxes = bboxes.reshape((-1, 5))
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        raise ValueError(
            'Invalid flipping direction "{}"'.format(direction))
    rotated_flag = (bboxes[:, 4] != -np.pi / 2)
    flipped[rotated_flag, 4] = -np.pi / 2 - bboxes[rotated_flag, 4]
    flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
    flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    return flipped.reshape(orig_shape)


def rbbox_mapping_back(bboxes,
                       img_shape,
                       scale_factor,
                       flip,
                       offset=(0., 0.),
                       flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = rbbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes.clone()
    if hasattr(scale_factor, "__len__"):
        w_scale, h_scale, _, _ = scale_factor
        wh_scale = np.sqrt(w_scale * h_scale)
    else:
        w_scale = scale_factor
        h_scale = scale_factor
        wh_scale = scale_factor
    new_bboxes[:, 0::5] /= w_scale
    new_bboxes[:, 1::5] /= h_scale
    new_bboxes[:, 2::5] /= wh_scale
    new_bboxes[:, 3::5] /= wh_scale
    new_bboxes[:, 0::5] += offset[0]
    new_bboxes[:, 1::5] += offset[1]
    return new_bboxes


def rbbox2points(rbboxes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5), xywha encoded

    Returns:
        bboxes (Tensor): shape (n, 8), x1y1x2y2x3y3x4y4
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)
