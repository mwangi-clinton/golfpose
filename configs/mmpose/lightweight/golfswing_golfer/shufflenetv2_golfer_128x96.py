# ShuffleNetV2 backbone — 128x96 ultra-low-resolution top-down
# Smallest practical config for the most efficient backbone.

_base_ = ['../_base_/default_runtime.py']

checkpoint_url = 'https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192-0aba71c7_20200921.pth'

train_cfg = dict(max_epochs=200, val_interval=10)

optim_wrapper = dict(optimizer=dict(
    type='Adam', lr=5e-4,
))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=200, milestones=[40, 60, 70],
         gamma=0.1, by_epoch=True),
]

auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater', max_keep_ckpts=3),
    early_stopping=dict(
        type='EarlyStoppingHook', monitor='coco/AP', patience=10,
        rule='greater', min_delta=0.001,
    ),
    visualization=dict(type='PoseVisualizationHook', enable=True, interval=10),
)

codec = dict(
    type='MSRAHeatmap', input_size=(128, 96), heatmap_size=(32, 24), sigma=2)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ShuffleNetV2',
        widen_factor=1.,
        out_indices=(3, ),
        init_cfg=dict(type='Pretrained',
                      checkpoint='mmcls://shufflenet_v2'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=22,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True, flip_mode='heatmap', shift_heatmap=True,
    ),
)

dataset_type = 'CocoDataset'
metainfo = dict(from_file='configs/mmpose/_base_/datasets/golfswing_golfer.py')
data_mode = 'topdown'
data_root = 'golfswing/'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=256, num_workers=32, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_golfer_2d_train.json',
        data_prefix=dict(img='images/'), metainfo=metainfo,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=256, num_workers=32, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_golfer_2d_test.json',
        data_prefix=dict(img='images/'), metainfo=metainfo,
        test_mode=True, pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'coco/hscc_golf_golfer_2d_test.json')
test_evaluator = val_evaluator


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='golfpose', name='golfer_shufflenetv2_golfer_128x96'))
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='golfer_shufflenetv2_golfer_128x96')
