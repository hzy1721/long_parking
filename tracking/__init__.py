import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from reid import config_path as fastreid_config_path, checkpoint_path as fastreid_checkpoint_path
from .utils.parser import get_config
from deep_sort import build_tracker
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

cfg = get_config()
cfg.merge_from_file(f'{ROOT}/configs/deep_sort.yaml')
cfg.DEEPSORT.REID_CKPT = f'{ROOT}/deep_sort/deep/checkpoint/ckpt.t7'
cfg.merge_from_file(f'{ROOT}/configs/fastreid.yaml')
cfg.USE_FASTREID = True
cfg.FASTREID.CFG = fastreid_config_path
cfg.FASTREID.CHECKPOINT = fastreid_checkpoint_path
deepsort = None


def init_track_model(device):
    global deepsort
    deepsort = build_tracker(cfg, use_cuda=torch.cuda.is_available(), device=device)
    logger.info(f'车辆跟踪模型加载完成 ({device})')


def track_vehicle(img, dets):
    """跟踪车辆

    :param img: ndarray
    :param dets: [[x1, y1, x2, y2, conf]]
    :return: [[x1, y1, x2, y2, track_id]]
    """
    if dets:
        dets = np.array(dets, dtype='float64')
        bbox, conf = dets[:, :4].astype('int32'), dets[:, 4]
        bbox_xywh = np.hstack((
            ((bbox[:, 0] + bbox[:, 2]) / 2).reshape(-1, 1),
            ((bbox[:, 1] + bbox[:, 3]) / 2).reshape(-1, 1),
            (bbox[:, 2] - bbox[:, 0]).reshape(-1, 1),
            (bbox[:, 3] - bbox[:, 1]).reshape(-1, 1)
        ))
        tracks = deepsort.update(bbox_xywh, conf, img)
        final_tracks = np.array(tracks).tolist()
    else:
        final_tracks = []
    logger.debug(f'tracks: {final_tracks}')
    return final_tracks


def update_tracker_info(tracker_info, tracks, plates):
    for *_, track_id in tracks:
        if track_id not in tracker_info:
            tracker_info[track_id] = {'plates': {}, 'lost_frames': 0}
    for track_id, *_, plate in plates:
        if plate not in tracker_info[track_id]['plates']:
            tracker_info[track_id]['plates'][plate] = 1
        else:
            tracker_info[track_id]['plates'][plate] += 1


def get_lost_track(tracker_info, tracks):
    """

    :param tracker_info:
    :param tracks:
    :return: [[track_id, { plate1: 1, plate2: 2 }]]
    """
    prev_track_ids = set(tracker_info.keys())
    curr_track_ids = set(np.ravel(np.array(tracks)[:, 4])) if tracks else set()
    lost_track_ids = prev_track_ids - curr_track_ids
    lost_track_plates = []
    lost_thresh = 25
    for track_id in lost_track_ids:
        tracker_info[track_id]['lost_frames'] += 1
        if tracker_info[track_id]['lost_frames'] > lost_thresh:
            if tracker_info[track_id]['plates']:
                lost_track_plates.append([track_id, tracker_info[track_id]['plates']])
            del tracker_info[track_id]
    return lost_track_plates


__all__ = ['track_vehicle', 'update_tracker_info', 'get_lost_track']
