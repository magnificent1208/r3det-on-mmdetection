# %%
import mmcv
import torch
import math
import matplotlib.pyplot as plt
import matplotlib
from mmdet.models.dense_heads import RRetinaHead, RRetinaRefineHead
from mmdet.core.bbox import rbbox2circumhbbox

matplotlib.use('Qt5Agg')


def drawrbbox(fig, rbboxes, color='r', lw=0.1):
    ax = fig.add_subplot(111)
    for rbbox in rbboxes:
        xc, yc, w, h, ag = rbbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        rect = plt.Rectangle((xc - wx - hx, yc - wy - hy), w, h, angle=ag / math.pi * 180, linewidth=lw,
                             facecolor='none', edgecolor=color)
        ax.add_patch(rect)
        ax.plot([xc], [yc], '+', c=color)
    ax.axis('equal')
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 640)
    ax.invert_yaxis()


train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        min_pos_iou=0.3,
        ignore_iof_thr=-1,
        iou_calculator=dict(type='RBboxOverlaps2D')
    ),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)
train_cfg = mmcv.Config(train_cfg)

num_cls = 2

rh = RRetinaHead(
    num_classes=num_cls,
    in_channels=256,
    train_cfg=train_cfg
)

rrh = RRetinaRefineHead(
    num_classes=num_cls,
    in_channels=256,
    train_cfg=train_cfg
)

###########################################
gt_bbox = torch.Tensor(
    [
        [300., 90., 100., 20., -math.pi / 4],
        [70., 80., 20., 100., -math.pi / 4],
        [600., 300., 130., 130., -math.pi / 12],
        [500., 300., 30., 5., -math.pi * 3 / 8],
        [250., 450., 250., 200., -math.pi * 4 / 9],
        [600., 450., 600., 300., -math.pi * 2 / 9]
    ]
).cuda()

gt_label = torch.Tensor(
    [
        0,
        1,
        0,
        0,
        1,
        0
    ]).type(torch.long).cuda()
gt_bbox = [gt_bbox, gt_bbox]
gt_label = [gt_label, gt_label]
###########################################
img_meta = [{
    'pad_shape': (640, 800, 3),  # (H, W, C)
    'img_shape': (640, 800, 3),
}] * 2  # 2 images
###########################################
feat_size = [(80, 100), (40, 50), (20, 25), (10, 13), (5, 7)]

###########################################
# code to test anchor generation, target assignment and sampling functions
###########################################
a, av = rh.get_anchors(feat_size, img_meta)
print("anchor_size")
for img_id in range(len(img_meta)):
    for lvl in range(len(feat_size)):
        print(a[img_id][lvl].size(), av[img_id][lvl].size())
###########################################

rt = rh.get_targets(anchor_list=a,
                    valid_flag_list=av,
                    gt_bboxes_list=gt_bbox,
                    img_metas=img_meta,
                    gt_labels_list=gt_label)
(labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
 num_total_pos, num_total_neg) = rt

a, av = rh.get_anchors(feat_size, img_meta)
# %%
fig = plt.figure(dpi=300)
drawrbbox(fig, gt_bbox[0], 'b', 2)
va_0 = []
vb_0 = []
for lvl in range(len(feat_size)):
    valid_a = a[0][lvl][labels_list[lvl][0] < num_cls]
    valid_bbt = bbox_targets_list[lvl][0][labels_list[lvl][0] < num_cls]
    va_0.append(valid_a)
    vb_0.append(valid_bbt)

va_0 = torch.cat(va_0)
vb_0 = torch.cat(vb_0)

drawrbbox(fig, va_0, 'r')
tb_0 = rh.bbox_coder.decode(va_0, vb_0)
# drawrbbox(fig, gt_bbox[0], 'cyan')
drawrbbox(fig, tb_0, 'm')
drawrbbox(fig, rbbox2circumhbbox(gt_bbox[0]), 'k')
plt.show()

###########################################
# code to test bbox filtering functions
###########################################
cls_scores = [torch.rand(
    (len(img_meta), rh.num_anchors * rh.cls_out_channels) + fs, device='cuda'
) for fs in feat_size]
bbox_pred = [torch.randn(
    (len(img_meta), rh.num_anchors * 5) + fs, device='cuda'
) for fs in feat_size]
# --------------------------------
filtered_bboxes = rh.filter_bboxes(cls_scores, bbox_pred)
print("filtered_bboxes_size")
for img_id in range(len(img_meta)):
    for lvl in range(len(feat_size)):
        print(filtered_bboxes[img_id][lvl].size())
###########################################
cls_scores = [torch.rand(
    (len(img_meta), rrh.num_anchors * rrh.cls_out_channels) + fs, device='cuda'
) for fs in feat_size]
bbox_pred = [torch.randn(
    (len(img_meta), rrh.num_anchors * 5) + fs, device='cuda'
) for fs in feat_size]
# --------------------------------
rrh.bboxes_as_anchors = filtered_bboxes
a, av = rrh.get_anchors(feat_size, img_meta)
rt = rrh.get_targets(a, av, gt_bbox, img_meta)
(labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
 num_total_pos, num_total_neg) = rt

plt.ion()
fig = plt.figure(dpi=300)
drawrbbox(fig, gt_bbox[0], 'b')
for lvl in range(len(feat_size)):
    valid_s = a[0][lvl][labels_list[lvl][0] < num_cls]
    for vs in valid_s:
        drawrbbox(fig, vs[None, :], 'r')
        plt.pause(0.5)
plt.ioff()

refined_bboxes = rrh.refine_bboxes(cls_scores, bbox_pred, a)
###########################################
# %%

plt.ion()
fig = plt.figure(dpi=300)
for img_id in range(len(img_meta)):
    for lvl in range(len(feat_size)):
        plt.clf()
        drawrbbox(fig, refined_bboxes[img_id][lvl][:500], 'g')
        plt.pause(0.5)
plt.ioff()


rrrh = RRetinaHead(11, 7)
x = torch.rand(1, 7, 32, 32)
cls_score, bbox_pred = rrrh.forward_single(x)
# Each anchor predicts a score for each class except background
cls_per_anchor = cls_score.shape[1] / rrrh.num_anchors
box_per_anchor = bbox_pred.shape[1] / rrrh.num_anchors
assert cls_per_anchor == rrrh.num_classes, (cls_per_anchor, rrrh.num_classes)
assert box_per_anchor == 5, box_per_anchor
