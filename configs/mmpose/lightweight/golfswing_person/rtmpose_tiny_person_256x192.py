# RTMPose-Tiny (CSPNeXt-Tiny backbone, ~4M params) — Top-down on GolfSwing person
# Fastest lightweight model; best for real-time inference.

_base_ = ['../_base_/default_runtime.py']

# Checkpoint: COCO-pretrained RTMPose-Tiny
checkpoint_url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth'

# runtime
train_cfg = dict(max_epochs=200, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='AdamW',
    lr=5e-4,
    weight_decay=0.05,
))

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=0.001, by_epoch=False),
    dict(type='CosineAnnealingLR', begin=0, end=200, eta_min=1e-6, by_epoch=True),
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

# codec
codec = dict(
    type='SimCCLabel', input_size=(256, 192), sigma=(5.66, 4.75),
    simcc_split_ratio=2.0, normalize=False, use_dark=False,
)

# model
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.',
                      checkpoint=checkpoint_url),
    ),
    head=dict(
        type='RTMCCHead',
        in_channels=384,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 6),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0., drop_path=0.,
            act_fn='SiLU', use_rel_bias=False, pos_enc=False,
        ),
        loss=dict(type='KLDiscretLoss', use_target_weight=True,
                  beta=10., label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True),
)

# dataset
dataset_type = 'CocoDataset'
metainfo = dict(from_file='configs/mmpose/_base_/datasets/golfswing_person.py')
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
    batch_size=256,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_person_2d_train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo, pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=256,
    num_workers=32,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_person_2d_test.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo, test_mode=True, pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'coco/hscc_golf_person_2d_test.json')
test_evaluator = val_evaluator


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='golfpose', name='person_rtmpose_tiny_golfer_256x192', group='golfswing_person'))
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='person_rtmpose_tiny_golfer_256x192')
