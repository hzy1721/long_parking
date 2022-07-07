import sys
sys.path.append('reid')

import os.path as osp
import cv2
import torch
import torch.nn.functional as F
import logging

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

__all__ = ['compute_feat']

logger = logging.getLogger(__name__)
reid_device = f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu'


def setup_cfg():
    cfg = get_cfg()
    reid_root_dir = osp.abspath(osp.dirname(__file__))
    cfg.merge_from_file(osp.join(reid_root_dir, 'configs/VeRi/sbs_R50-ibn.yml'))
    cfg.merge_from_list(['MODEL.WEIGHTS', osp.join(reid_root_dir, 'checkpoints/veri_sbs_R50-ibn.pth'),
                         'MODEL.DEVICE', reid_device])
    cfg.freeze()
    return cfg


cfg = setup_cfg()
# predictor = DefaultPredictor(cfg)
predictor = None
# logger.info(f'车辆重识别模型加载完成 ({reid_device})')


def postprocess(features):
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def compute_feat(img_car):
    img_car = img_car[:, :, ::-1]
    img = cv2.resize(img_car, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]
    feat = predictor(img)
    feat = postprocess(feat)
    return feat[0]
