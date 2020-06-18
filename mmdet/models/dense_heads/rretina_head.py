import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from . import AnchorHead

from mmdet.core import (ranchor_inside_flags, force_fp32, multiclass_rnms,
                        unmap, rbbox2circumhbbox)

from ..builder import HEADS


@HEADS.register_module
class RRetinaHead(AnchorHead):
    """Rotational Anchor-based head

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_h_gt=True,
                 anchor_generator=dict(
                     type='RAnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 0.2, 5.0],
                     angles=None,
                     strides=[8, 16, 32, 64, 128]),
                 bbox_coder=dict(
                     type='DeltaXYWHABBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_h_gt = use_h_gt
        if use_h_gt:
            angles = anchor_generator["angles"]
            assert angles is None or (torch.tensor(angles).numel() == 1 and torch.all(torch.tensor(angles) == 0))

        super(RRetinaHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 5)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = ranchor_inside_flags(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.use_h_gt:
            anchors_assign = rbbox2circumhbbox(anchors)
            gt_bboxes_assign = rbbox2circumhbbox(gt_bboxes)
            if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
                gt_bboxes_ignore_assign = rbbox2circumhbbox(gt_bboxes_ignore)
            else:
                gt_bboxes_ignore_assign = None
        else:
            anchors_assign = anchors
            gt_bboxes_assign = gt_bboxes
            gt_bboxes_ignore_assign = gt_bboxes_ignore

        assign_result = self.assigner.assign(
            anchors_assign, gt_bboxes_assign, gt_bboxes_ignore_assign,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self,
                      cls_scores,
                      bbox_preds):
        """
                Filter predicted bounding boxes at each position of the feature maps.
                Only one bounding boxes with highest score will be left at each position.
                This filter will be used in R3Det prior to the first feature refinement stage.

                Args:
                    cls_scores (list[Tensor]): Box scores for each scale level
                        Has shape (N, num_anchors * num_classes, H, W)
                    bbox_preds (list[Tensor]): Box energies / deltas for each scale
                        level with shape (N, num_anchors * 5, H, W)

                Returns:
                    list[list[Tensor]]: best or refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]  # list(indexed by images) of list(indexed by levels)

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)  # (N, H, W, A*C)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors, self.cls_out_channels)  # (N, H*W, A, C)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)  # (N, H*W, A, 1)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)  # (N, H*W, 1, 1)
            best_ind = best_ind.expand(-1, -1, -1, 5)  # (N, H*W, 1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)  # (N, H, W, A*5)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)  # (N, H*W, A, 5)

            best_pred = bbox_pred.gather(dim=-2, index=best_ind).squeeze(dim=-2)  # (N, H*W, 5)

            # anchors shape (H*W*A, 5)
            anchors = anchors.reshape(-1, self.num_anchors, 5)  # (H*W, A, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]  # (H*W, 1, 5)
                best_pred_i = best_pred[img_id]  # (H*W, 5)
                best_anchor_i = anchors.gather(dim=-2, index=best_ind_i).squeeze(dim=-2)  # (H*W, 5)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i, best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """
        Transform outputs for a single batch item (a single image) into labeled boxes.

        Args:
            cls_score_list (list[Tensor]): Box scores for each scale level of a single image
                Has shape (num_anchors * num_classes, H, W)
            bbox_pred_list (list[Tensor]): Box energies / deltas for each scale of a single image
                Has shape (num_anchors * 5, H, W)
            mlvl_anchors (list[Tensor]): Anchors for each scale of a single image
                Has shape (H * W * num_anchors, 5)
            img_shape tuple[int, int]: Image size
            scale_factor float: Image scale factor
            cfg (mmcv.Config): test / postprocessing configuration
            rescale (bool): if True, return boxes in original image space
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,
                                                 cfg.score_thr, cfg.nms,
                                                 cfg.max_per_img)
        return det_bboxes, det_labels
