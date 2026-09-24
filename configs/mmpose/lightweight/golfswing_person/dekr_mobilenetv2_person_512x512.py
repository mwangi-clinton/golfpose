# DEKR + MobileNetV2 backbone — Bottom-up on GolfSwing person
# Bottom-up multi-person approach using disentangled keypoint regression
# with a lightweight MobileNetV2 backbone.

_base_ = ['../_base_/default_runtime.py']

# DEKR doesn't have a MobileNetV2 official checkpoint — train from scratch
# with MobileNetV2 ImageNet-pretrained backbone.
checkpoint_url = None

train_cfg = dict(max_epochs=200, val_interval=10)

optim_wrapper = dict(optimizer=dict(
    type='Adam', lr=1e-3,
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

# Bottom-up DEKR codec
codec = dict(
    type='SPR',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=(4, 2),
    minimal_diagonal_length=32,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30,
)

model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(2, 4, 7),  # Multi-scale outputs for bottom-up
        init_cfg=dict(type='Pretrained',
                      checkpoint='mmcls://mobilenet_v2'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='DEKRHead',
        in_channels=1280,  # Adjusted for MobileNetV2 concat
        num_keypoints=17,
        num_heatmap_filters=32,
        num_offset_filters_per_kpt=15,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1/9.,
            loss_weight=0.002,
        ),
        decoder=codec,
        rescore_cfg=dict(
            in_channels=74,
            norm_indexes=(5, 6),
        ),
    ),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False,
    ),
)

dataset_type = 'CocoDataset'
metainfo = dict(from_file='configs/mmpose/_base_/datasets/golfswing_person.py')
data_mode = 'bottomup'
data_root = 'golfswing/'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupResize', input_size=codec['input_size'],
         resize_mode='fit', pad_val=(114, 114, 114)),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=256, num_workers=32, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_person_2d_train.json',
        data_prefix=dict(img='images/'), metainfo=metainfo,
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=256, num_workers=32, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='coco/hscc_golf_person_2d_test.json',
        data_prefix=dict(img='images/'), metainfo=metainfo,
        test_mode=True, pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'coco/hscc_golf_person_2d_test.json')
test_evaluator = val_evaluator


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='golfpose', name='person_dekr_mobilenetv2_golfer_512x512'))
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='person_dekr_mobilenetv2_golfer_512x512')
