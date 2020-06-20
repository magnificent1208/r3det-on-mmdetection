import mmcv
import torch
import numpy as np
from mmdet.core import tensor2imgs


def image_merge(data):
    img = data['img']
    img_metas = data['img_metas']
    h, w, = 0, 0

    for i in range(len(img)):
        img_meta = img_metas[i].data[0]
        assert len(img_meta) == 1
        img_meta = img_meta[0]
        x_off, y_off = img_meta['tile_offset']
        h_tile, w_tile, _ = img_meta['pad_shape']
        x_scale, y_scale = tuple(img_meta['scale_factor'])[:2]

        h_tile = int(round(h_tile / y_scale))
        w_tile = int(round(w_tile / x_scale))
        h = max(h, y_off + h_tile)
        w = max(w, x_off + w_tile)

    img_show = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(img)):
        img_tile = data['img'][i]
        img_meta = img_metas[i].data[0]
        assert len(img_tile) == len(img_meta) == 1
        img_tile = img_tile
        img_meta = img_meta[0]
        flip = img_meta['flip']
        flip_dir = img_meta['flip_direction']
        x_off, y_off = img_meta['tile_offset']
        h_tile, w_tile, _ = img_meta['pad_shape']
        x_scale, y_scale = tuple(img_meta['scale_factor'])[:2]

        h_tile = int(round(h_tile / y_scale))
        w_tile = int(round(w_tile / x_scale))

        img_tile = tensor2imgs(img_tile, **img_meta['img_norm_cfg'])[0]
        img_tile = mmcv.imresize(img_tile, (h_tile, w_tile))
        if flip:
            img_tile = mmcv.imflip(img_tile, direction=flip_dir)
        img_show[y_off: y_off + h_tile, x_off:x_off + w_tile, :] = img_tile

    return img_show


def single_gpu_mergetiles_visualize(model,
                                    data_loader,
                                    show_score_thr=0.3):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        img_show = image_merge(data)

        model.module.show_result(
            img_show,
            result,
            show=True,
            out_file=None,
            score_thr=show_score_thr)

        prog_bar.update()
