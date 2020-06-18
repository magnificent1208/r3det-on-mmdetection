from mmcv import Config
import numpy as np
from mmdet.datasets import build_dataset, build_dataloader
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.use('Qt5Agg')


def draw(fig, img, boxes, ignore_bboxes):
    ax = fig.add_subplot(111)
    # ax.set_xlim(0, 800)
    # ax.set_ylim(0, 800)
    ax.invert_yaxis()
    ax.imshow(img)
    for rbbox in boxes:
        xc, yc, w, h, ag = rbbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        rect = plt.Rectangle((xc - wx - hx, yc - wy - hy), w, h, angle=ag / math.pi * 180, linewidth=0.5,
                             facecolor='none', edgecolor='b')
        ax.add_patch(rect)
        ax.plot([xc], [yc], '+', c='b', markersize=0.5)

    for rbbox in ignore_bboxes:
        xc, yc, w, h, ag = rbbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        rect = plt.Rectangle((xc - wx - hx, yc - wy - hy), w, h, angle=ag / math.pi * 180, linewidth=1,
                             facecolor='none', edgecolor='r')
        ax.add_patch(rect)
        ax.plot([xc], [yc], '+', c='g', markersize=1)


cfg = Config.fromfile('../configs/r3det/r3det_r50_fpn_2x.py')

datasets = build_dataset(cfg.data.train)

data_loader = build_dataloader(datasets,
                               6,
                               0,
                               dist=False)

for data in data_loader:
    print(data.keys())
    img_metas = data['img_metas'].data[0]
    img = data['img'].data[0]
    gt_bboxes = data['gt_bboxes'].data[0]
    gt_bboxes_ignore = data['gt_bboxes_ignore'].data[0]
    for i in range(len(img_metas)):
        fig = plt.figure(dpi=300)
        mean = img_metas[i]['img_norm_cfg']['mean']
        std = img_metas[i]['img_norm_cfg']['std']
        img_show = img[i].permute(1, 2, 0) * std[None, None, :] + mean[None, None, :]
        img_show = np.clip(img_show.numpy().astype(np.int), 0, 255)
        draw(fig, img_show, gt_bboxes[i], gt_bboxes_ignore[i])
        plt.show()

    pass
