# Runtime settings shared by every DADNet config.
# `default_scope` is required: DADNet is registered in the `mmseg` registry
# scope, so without it mmengine looks for `EncoderDecoder` in its own root
# registry and fails with "EncoderDecoder is not in the mmengine::model
# registry".
default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
