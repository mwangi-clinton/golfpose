#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-lightweight golfer + club detector using RTMDet-nano.

RTMDet-nano: ~1.6M params, ~2.4 GFLOPs @ 320x320
This is the FASTEST option in the RTMDet family.
Directly comparable to MediaPipe's BlazeFace person detector (~2M params).

Architecture summary:
  - Backbone : CSPNeXt-nano (minimal depth/width multipliers)
  - Neck     : CSPNeXtPAFPN-nano (single CSP block per scale)
  - Head     : RTMDetSepBNHead (anchor-free, decoupled)
  - Input    : 320x320

Key difference from RTMDet-tiny:
  - deepen_factor: 0.167 → same depth
  - widen_factor : 0.375 → 0.25 (fewer channels = fewer FLOPs and params)
"""

default_scope = 'mmdet'

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

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Download: mim download mmdet --config rtmdet_nano_8xb32-300e_coco --dest models/
load_from = 'models/rtmdet_nano_8xb32-300e_coco_20220902_111946-2986f400.pth'
resume = False

max_epochs = 100
num_last_epochs = 10

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

base_lr = 0.004

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0, end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=35, norm_type=2),
)

auto_scale_lr = dict(enable=True, base_batch_size=32)

INPUT_SIZE = (320, 320)

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
        deepen_factor=0.167,   # same as tiny
        widen_factor=0.25,     # NARROWER than tiny (0.375) — key difference
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=load_from),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[64, 128, 256],   # narrower channels than tiny [96,192,384]
        out_channels=64,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=2,
        in_channels=64,        # matches narrower neck output
        stacked_convs=2,
        feat_channels=64,
        anchor_generator=dict(type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=300,
        min_bbox_size=0,
        score_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=10,
    ),
)

dataset_type = 'CocoDataset'
data_root    = 'golfswing/'
classes      = ('person', 'club')
backend_args = None

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
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/hscc_golf_det_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline_stage1,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='coco/hscc_golf_det_test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/hscc_golf_det_test.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

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
