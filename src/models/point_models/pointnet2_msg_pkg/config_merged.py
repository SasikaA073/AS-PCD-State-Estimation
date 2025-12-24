class_names = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
               'bathtub', 'otherfurniture')
metainfo = dict(
    classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
             'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
             'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
             'bathtub', 'otherfurniture'))
dataset_type = 'ScanNetSegDataset'
data_root = 'data/scannet/'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')
backend_args = None
num_points = 8192
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=None),
    dict(type='PointSegClassMapping'),
    dict(
        type='IndoorPatchPointSample',
        num_points=8192,
        block_size=1.5,
        ignore_index=20,
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'RandomFlip3D',
            'sync_2d': False,
            'flip_ratio_bev_horizontal': 0.0,
            'flip_ratio_bev_vertical': 0.0
        }], [{
            'type': 'Pack3DDetInputs',
            'keys': ['points']
        }]])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ScanNetSegDataset',
        data_root='data/scannet/',
        ann_file='scannet_infos_train.pkl',
        metainfo=dict(
            classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture',
                     'counter', 'desk', 'curtain', 'refrigerator',
                     'showercurtrain', 'toilet', 'sink', 'bathtub',
                     'otherfurniture')),
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5],
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True,
                backend_args=None),
            dict(type='PointSegClassMapping'),
            dict(
                type='IndoorPatchPointSample',
                num_points=8192,
                block_size=1.5,
                ignore_index=20,
                use_normalized_coord=False,
                enlarge_size=0.2,
                min_unique_num=None),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=20,
        scene_idxs='data/scannet/seg_info/train_resampled_scene_idxs.npy',
        test_mode=False,
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ScanNetSegDataset',
        data_root='data/scannet/',
        ann_file='scannet_infos_val.pkl',
        metainfo=dict(
            classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture',
                     'counter', 'desk', 'curtain', 'refrigerator',
                     'showercurtrain', 'toilet', 'sink', 'bathtub',
                     'otherfurniture')),
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5],
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True,
                backend_args=None),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=20,
        test_mode=True,
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ScanNetSegDataset',
        data_root='data/scannet/',
        ann_file='scannet_infos_val.pkl',
        metainfo=dict(
            classes=('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture',
                     'counter', 'desk', 'curtain', 'refrigerator',
                     'showercurtrain', 'toilet', 'sink', 'bathtub',
                     'otherfurniture')),
        data_prefix=dict(
            pts='points',
            pts_instance_mask='instance_mask',
            pts_semantic_mask='semantic_mask'),
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5],
                backend_args=None),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_mask_3d=False,
                with_seg_3d=True,
                backend_args=None),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(type='Pack3DDetInputs', keys=['points'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        ignore_index=20,
        test_mode=True,
        backend_args=None))
val_evaluator = dict(type='SegMetric')
test_evaluator = dict(type='SegMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
tta_model = dict(type='Seg3DTTAModel')
model = dict(
    type='EncoderDecoder3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=6,
        num_points=(1024, 256, 64, 16),
        radii=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
        num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
        aggregation_channels=(None, None, None, None),
        fps_mods=('D-FPS', 'D-FPS', 'D-FPS', 'D-FPS'),
        fps_sample_range_lists=(-1, -1, -1, -1),
        dilated_group=(False, False, False, False),
        out_indices=(0, 1, 2, 3),
        sa_cfg=dict(
            type='PointSAModuleMSG',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    decode_head=dict(
        type='PointNet2Head',
        fp_channels=((1536, 256, 256), (512, 256, 256), (352, 256, 128),
                     (128, 128, 128, 128)),
        channels=128,
        dropout_ratio=0.5,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[
                2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
                4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
                5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
                5.3954206, 4.6971426
            ],
            loss_weight=1.0),
        num_classes=20,
        ignore_index=20),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.01),
    clip_grad=None)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=1e-05,
        by_epoch=True,
        begin=0,
        end=200)
]
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=5)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=32)
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
