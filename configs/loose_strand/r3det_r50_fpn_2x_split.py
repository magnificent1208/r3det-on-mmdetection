#_base_ = [
#    '../../configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py'
#]

# model settings
model = dict(
    type='R3Det',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHead',
        num_classes=1, # CLASS
        in_channels=256,
        stacked_convs=4,
        use_h_gt=True,
        feat_channels=256,
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
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=0.11,
            loss_weight=1.0)),
    frm_cfgs=[
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128])
    ],
    num_refine_stages=2,
    refine_heads=[
        dict(
            type='RRetinaRefineHead',
            num_classes=1, # CLASS
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
        dict(
            type='RRetinaRefineHead',
            num_classes=1, # CLASS
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
    ]
)

# training and testing settings
train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    sr=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.6,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        )
    ],
    stage_loss_weights=[1.0, 1.0]
)

merge_nms_iou_thr_dict = {
    'loose_strand': 0.1
}

merge_cfg = dict(
    nms_pre=2000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=merge_nms_iou_thr_dict),
    max_per_img=1000,
)

test_cfg = dict(
    nms_pre=1000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=0.05),
    max_per_img=100,
    merge_cfg=merge_cfg
)



# dataset settings
dataset_type = 'LOOSE_STRAND_R3D'
classes = ('loose_strand')
# dataset root path:
data_root = 'data/loose_strand/'
trainsplit_ann_folder = 'trainsplit/labelTxt'
trainsplit_img_folder = 'trainsplit/images'
valsplit_ann_folder = 'valsplit/labelTxt'
valsplit_img_folder = 'valsplit/images'
val_ann_folder = 'val/labelTxt'
val_img_folder = 'val/images'
test_img_folder = 'val/images'

img_norm_cfg = dict(
    mean=[131.6437546, 134.80090441, 120.2500286], 
    std=[40.86244546, 38.11052274, 43.29354337], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CroppedTilesFlipAug',
        tile_scale=(800, 800),
        tile_shape=(600, 600),
        tile_overlap=(300, 300),
        flip=False,
        transforms=[
            dict(type='RResize', img_scale=(800, 800)),
            dict(type='RRandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(800, 800)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=trainsplit_ann_folder,
            img_prefix=trainsplit_img_folder,
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=valsplit_ann_folder,
            img_prefix=valsplit_img_folder,
            pipeline=train_pipeline),
    ],
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=val_ann_folder,
            img_prefix=val_img_folder,
            pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_img_folder,
        img_prefix=test_img_folder,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD',
                 lr=4e-3,
                 momentum=0.9,
                 weight_decay=0.0001,
                 )  # paramwise_options=dict(bias_lr_mult=2, bias_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.1,
    step=[12, 16, 20])
total_epochs = 24



# runtime settings
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/loose_strand/r3det_r50_fpn_2x_split_210726'
evaluation = dict(interval=1, metric='mAP')