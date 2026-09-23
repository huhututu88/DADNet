# DADNet (UNet32) on the NEU-Seg dataset.
#
#   python tools/train.py configs/dadnet/dadnet_unet32_neu_seg.py
#   python tools/test.py  configs/dadnet/dadnet_unet32_neu_seg.py \
#       work_dirs/dadnet_unet32_neu_seg/best_mIoU_iter_XXXX.pth
_base_ = [
    '_base_/dadnet_unet32.py',
    '_base_/datasets/neu_seg.py',
    '_base_/default_runtime.py',
]

num_classes = 4  # background, In, Pa, Sc
model = dict(decode_head=dict(num_classes=num_classes))

# Adam, poly decay 0.9, 1e-4 -> 1e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, weight_decay=1e-5),
)
param_scheduler = [
    dict(type='PolyLR', begin=0, end=10000, power=0.9, eta_min=1e-5,
         by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100,
                    max_keep_ckpts=1, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

randomness = dict(seed=0, deterministic=False)
work_dir = './work_dirs/dadnet_unet32_neu_seg'
