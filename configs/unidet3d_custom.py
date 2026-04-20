"""
UniDet3D 自定义数据集训练配置
支持任意类别，只需修改 classes_custom 和数据路径。
"""
_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unidet3d'])

# ── 自定义类别（按需增减） ──────────────────────────────────────────────────────
classes_custom = ['car', 'chair', 'person']
num_classes_custom = len(classes_custom)   # 无 stuff 类，全部为 foreground

# ── 模型超参 ───────────────────────────────────────────────────────────────────
num_channels = 32
voxel_size   = 0.02

model = dict(
    type='UniDet3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=6,
    num_channels=num_channels,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr=3000,
    bbox_by_mask=[True],        # 训练时从 instance mask 动态计算 bbox
    target_by_distance=[False],
    use_superpoints=[True],
    fast_nms=[True],
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='UniDet3DEncoder',
        num_layers=6,
        datasets_classes=[classes_custom],   # ← 自定义类别
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        datasets=['custom'],                 # ← 数据集标签（与 lidar_path 对应）
        angles=[False]),
    criterion=dict(
        type='UniDet3DCriterion',
        datasets=['custom'],
        datasets_weights=[1],
        bbox_loss_simple=dict(
            type='UniDet3DAxisAlignedIoULoss', mode='diou', reduction='none'),
        bbox_loss_rotated=dict(
            type='UniDet3DRotatedIoU3DLoss', mode='diou', reduction='none'),
        matcher=dict(
            type='UniMatcher',
            costs=[
                dict(type='QueryClassificationCost', weight=0.5),
                dict(type='BboxCostJointTraining',
                     weight=2.0,
                     loss_simple=dict(
                         type='UniDet3DAxisAlignedIoULoss',
                         mode='diou', reduction='none'),
                     loss_rotated=dict(
                         type='UniDet3DRotatedIoU3DLoss',
                         mode='diou', reduction='none'))]),
        loss_weight=[0.5, 1.0],
        non_object_weight=0.1,
        topk=[6],
        iter_matcher=True),
    train_cfg=dict(topk=6),
    test_cfg=dict(
        low_sp_thr=0.18,
        up_sp_thr=0.81,
        topk_insts=1000,
        score_thr=0,
        iou_thr=[0.5]))

# ── 数据集配置 ────────────────────────────────────────────────────────────────
data_root   = 'data/custom/'
ann_prefix  = ''            # pkl 与 data_root 同级
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

# 训练 pipeline：与 ScanNet 基本相同，去掉 ScanNet 专属的 PointSegClassMapping
train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0,1,2,3,4,5]),
    dict(type='LoadAnnotations3D_',
         with_bbox_3d=False, with_label_3d=False,
         with_mask_3d=True, with_seg_3d=True, with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='RandomFlip3D',
         sync_2d=False,
         flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-3.14, 3.14],
         scale_ratio_range=[0.8, 1.2],
         translation_std=[0.1, 0.1, 0.1],
         shift_height=False),
    dict(type='NormalizePointsColor_',
         color_mean=[127.5, 127.5, 127.5]),
    # stuff_classes=[] 表示无背景类，所有类别均为 foreground
    dict(type='PointDetClassMappingScanNet',
         num_classes=num_classes_custom,
         stuff_classes=[]),
    dict(type='ElasticTransfrom',
         gran=[6, 20], mag=[40, 160], voxel_size=voxel_size, p=0.5),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
               'pts_instance_mask', 'sp_pts_mask', 'gt_sp_masks',
               'elastic_coords'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='DEPTH', shift_height=False,
         use_color=True, load_dim=6, use_dim=[0,1,2,3,4,5]),
    dict(type='LoadAnnotations3D_',
         with_bbox_3d=False, with_label_3d=False,
         with_mask_3d=True, with_seg_3d=True, with_sp_mask_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[dict(type='NormalizePointsColor_',
                          color_mean=[127.5, 127.5, 127.5])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomRGBDDetDataset',
        data_root=data_root,
        ann_file='custom_infos_train.pkl',
        metainfo=dict(classes=classes_custom),
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomRGBDDetDataset',
        data_root=data_root,
        ann_file='custom_infos_val.pkl',
        metainfo=dict(classes=classes_custom),
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True,
        ))

test_dataloader = val_dataloader

val_evaluator  = dict(type='IndoorMetric')
test_evaluator = val_evaluator

# ── 训练策略 ──────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(
    type='PolyLR', begin=0, end=512, power=0.9)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=512,
    val_interval=16)

val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=16, max_keep_ckpts=3))

# ── 从预训练模型加载 backbone 权重（可选） ────────────────────────────────────
# 注意：预训练模型与当前类别不同，只能加载 backbone，decoder 随机初始化
# 如需微调，取消注释：
# load_from = 'work_dirs/tmp/unidet3d.pth'