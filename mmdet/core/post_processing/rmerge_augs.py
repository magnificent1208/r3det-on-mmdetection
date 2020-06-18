import torch

from ..post_processing import aug_multiclass_rnms
from ..bbox import rbbox_mapping_back


def merge_tiles_aug_rbboxes(aug_bboxes,
                           aug_labels,
                           img_metas,
                           merge_cfg,
                           classes=None):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 5 + 1)
        aug_labels (list[Tensor] or None): shape (n,)
        img_metas (list[dict]): meta information of images.
        merge_cfg (dict): test config.
        classes (tuple): tuple of classes names in order.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        tile_offset = img_info[0]['tile_offset']
        bboxes[:, :-1] = rbbox_mapping_back(bboxes[:, :-1],
                                            img_shape,
                                            scale_factor,
                                            flip,
                                            tile_offset)
        recovered_bboxes.append(bboxes)
    global_bboxes = torch.cat(recovered_bboxes)
    global_labels = torch.cat(aug_labels)

    nms_pre = merge_cfg.get('nms_pre', -1)
    if nms_pre > 0 and global_bboxes.shape[0] > nms_pre:
        _, topk_inds = global_bboxes[:, -1].topk(nms_pre)
        global_bboxes = global_bboxes[topk_inds, :]
        global_labels = global_labels[topk_inds]

    if isinstance(merge_cfg.nms['iou_thr'], dict):
        assert classes is not None
        iou_thr_dict = merge_cfg.nms['iou_thr']
        merge_cfg.nms['iou_thr'] = [*map(lambda cls: iou_thr_dict.get(cls, 0.05), classes)]

    global_bboxes, global_labels = aug_multiclass_rnms(
        global_bboxes,
        global_labels,
        merge_cfg.score_thr,
        merge_cfg.nms,
        merge_cfg.max_per_img)
    return global_bboxes, global_labels
