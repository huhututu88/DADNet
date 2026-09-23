# DADNet / UNet32 model definition, shared by every dataset config.
#
# feature_scale=2 divides the canonical U-Net widths (64, 128, 256, 512, 1024)
# by 2, giving a full-resolution width of 32 -- hence the "UNet32" tag used
# throughout the paper's tables.
#
# The four arguments that toggle the ablations are:
#   backbone.is_catch    -> DFWE   (dual-branch grouped encoding)
#   backbone.is_fusion   -> MDCF   (multi-scale difference convolution fusion)
#   backbone.kernel_sizes-> the (k1, k2) depth-wise kernel pair inside MDCF
#   backbone.pool_type   -> 'avg' (paper default) or 'max'

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 256),
)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DADNet',
        in_channels=3,
        feature_scale=2,
        is_batchnorm=True,
        is_deconv=True,
        norm_eval=False,
        # --- ablation switches (paper setting shown first) ---
        is_catch=True,          # DFWE
        is_fusion=True,         # MDCF
        kernel_sizes=(5, 7),    # MDCF depth-wise kernel pair
        pool_type='avg',        # down-sampling pooling
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=32,
        in_index=0,
        channels=32,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        init_cfg=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # num_classes is set by each dataset config
    ),
    pretrained=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)
