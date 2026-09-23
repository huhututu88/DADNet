# NEU-Seg (a.k.a. SD900) hot-rolled steel strip surface defect dataset.
# 4 classes: background, In (inclusion), Pa (patches), Sc (scratches).
#
# Layout expected under data_root:
#   data/NEU-Seg/
#   ├── img_dir/{train,val,test}/*.bmp
#   ├── ann_dir/{train,val,test}/*.png
#   └── {train,val,test}.json      # produced by tools/prepare_dataset.py
dataset_type = 'SD900Dataset'
data_root = 'data/NEU-Seg'

img_suffix = '.bmp'
seg_map_suffix = '.png'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='Random_Choices',
         operations=['RandomRotate_Choices', 'RandomFlip']),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False,
        data_prefix=dict(img_path='train.json', seg_map_path='train.json'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False,
        data_prefix=dict(img_path='val.json', seg_map_path='val.json'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        reduce_zero_label=False,
        data_prefix=dict(img_path='test.json', seg_map_path='test.json'),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
