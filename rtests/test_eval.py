from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.core.evaluation import reval_map

cfg = Config.fromfile('../configs/r3det/r3det_r50_fpn_2x.py')

dataset = build_dataset(cfg.data.val)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    dist=False,
    shuffle=False)

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

checkpoint = load_checkpoint(model, "../work_dirs/r3det_r50_fpn_2x_20200616/epoch_24.pth", map_location='cpu')

if 'CLASSES' in checkpoint['meta']:
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader)
dataset.evaluate(outputs)
