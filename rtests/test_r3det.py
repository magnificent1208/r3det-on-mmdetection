from mmcv import Config
import numpy as np
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
from mmdet.apis import train_detector

cfg = Config.fromfile('../configs/r3det/r3det_r50_fpn_2x.py')

datasets = build_dataset(cfg.data.train)

data_loader = build_dataloader(datasets,
                               6,
                               0,
                               dist=False)

model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model = MMDataParallel(model, device_ids=[0])

for data in data_loader:
    print(model(**data))
