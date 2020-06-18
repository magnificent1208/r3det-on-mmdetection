import torch

from mmdet.ops.nms import batched_rnms, rnms


def multiclass_rnms(multi_bboxes,
                    multi_scores,
                    score_thr,
                    nms_cfg,
                    max_num=-1,
                    score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 6))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    dets, keep = batched_rnms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def aug_multiclass_rnms(global_bboxes,
                        global_labels,
                        score_thr,
                        nms_cfg,
                        max_num=-1,
                        score_factors=None):
    classes = global_labels.unique()

    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_thr = nms_cfg_.pop('iou_thr', 0.05)
    nms_op = eval(nms_type)

    for cls in classes:
        cls_bboxes = global_bboxes[global_labels == cls]
        _inds = cls_bboxes[:, -1] > score_thr
        _bboxes = cls_bboxes[_inds, :]

        iou_thr = nms_thr[cls] if hasattr(nms_thr, '__getitem__') else nms_thr

        cls_dets, _ = nms_op(_bboxes, **nms_cfg_, iou_thr=iou_thr)
        cls_labels = global_bboxes.new_full((cls_dets.shape[0],),
                                            cls,
                                            dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = global_bboxes.new_zeros((0, global_bboxes.size(-1)))
        labels = global_bboxes.new_zeros((0,), dtype=torch.long)

    return bboxes, labels
