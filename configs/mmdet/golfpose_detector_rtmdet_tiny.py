#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-lightweight golfer + club detector using RTMDet-tiny.

RTMDet-tiny: ~4.8M params, ~8.1 GFLOPs @ 320x320
Comparable to MediaPipe's BlazeFace/SSD pipeline.
Far smaller than YOLOX-S (9M params) and YOLOX-nano (0.9M but lower accuracy).

Architecture summary:
  - Backbone : CSPNeXt-tiny (depthwise-separable convs, low channel count)
  - Neck     : CSPNeXtPAFPN (lightweight path-aggregation)
  - Head     : RTMDetSepBNHead (decoupled cls+reg, no anchors)
  - Input    : 320x320 (half of standard 640; 4x fewer pixels to process)

Training tips:
  - Pretrained weights from COCO significantly reduce convergence time
  - 100 epochs is sufficient for this dataset size (~14k images)
  - Mixed precision (fp16) halves GPU memory, run with --amp flag
"""

default_scope = 'mmdet'

# ─── Hooks ────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
    ),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# ─── Environment ──────────────────────────────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

# ─── Logging ──────────────────────────────────────────────────────────────────
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# ─── Pretrained weights (COCO RTMDet-tiny) ────────────────────────────────────
# Download once:  mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest models/
# Or set to None to train from scratch
load_from = 'models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
resume = False

# ─── Training schedule ────────────────────────────────────────────────────────
# 100 epochs with cosine annealing — matches RTMDet paper's recipe
max_epochs = 100
num_last_epochs = 10   # last N epochs: close mosaic, use regular aug

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# ─── Learning rate (cosine with warmup) ───────────────────────────────────────
base_lr = 0.004   # scaled for batch_size=32; adjust with auto_scale_lr

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# ─── Optimizer (AdamW — RTMDet default) ───────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

auto_scale_lr = dict(enable=True, base_batch_size=32)

# ─── Model: RTMDet-tiny @ 320×320 ─────────────────────────────────────────────
INPUT_SIZE = (320, 320)   # (H, W) — half the standard 640; 4x fewer FLOPs

model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,   # tiny depth multiplier
        widen_factor=0.375,    # tiny width multiplier
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=load_from,
        ),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=2,   # person, club
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=300,
        min_bbox_size=0,
        score_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=10,   # at most 10 detections per frame (1 golfer + 1 club + margin)
    ),
)

# ─── Dataset ──────────────────────────────────────────────────────────────────
dataset_type = 'CocoDataset'
data_root    = 'golfswing/'
classes      = ('person', 'club')
backend_args = None

# ─── Augmentation (RTMDet recipe) ─────────────────────────────────────────────
train_pipeline_stage1 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=INPUT_SIZE,
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
    ),
    dict(
        type='RandomResize',
        scale=(INPUT_SIZE[1] * 2, INPUT_SIZE[0] * 2),
        ratio_range=(0.5, 2.0),
        resize_type='Resize',
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=INPUT_SIZE),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=INPUT_SIZE, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

# Last `num_last_epochs` epochs: disable mosaic for cleaner convergence
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=INPUT_SIZE, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Pad', size=INPUT_SIZE, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=INPUT_SIZE, keep_ratio=True),
    dict(type='Pad', size=INPUT_SIZE, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# ─── DataLoaders ──────────────────────────────────────────────────────────────
train_batch_size = 32
val_batch_size   = 8
num_workers      = 4

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/hscc_golf_det_train.json',   # generated by extract_gt_bboxes.py
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline_stage1,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/hscc_golf_det_test.json',    # generated by extract_gt_bboxes.py
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# ─── Evaluator ────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/hscc_golf_det_test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# ─── Switch augmentation in last epochs ───────────────────────────────────────
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]
