# checkpoint_config = dict(interval=1) # interval=10 表示1个epoch保存一次
checkpoint_config = dict(interval=1) # interval=10 表示10个epoch保存一次
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
# 微调
# load_from = "work_dirs/faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
# load_from ="work_dirs/fcos_r101_fpn_3x_det_bdd100k.pth"

resume_from = None

workflow = [('train', 1)]
