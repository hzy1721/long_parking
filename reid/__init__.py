import numpy as np
from numpy.linalg import norm
import time
import os
from .compute_feat import compute_feat
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
config_path = f'{ROOT}/configs/VeRi/sbs_R50-ibn.yml'
checkpoint_path = f'{ROOT}/checkpoints/veri_sbs_R50-ibn.pth'


def cosine_distance(feat1, feat2):
    cosine_value = np.sum(feat1 * feat2) / (norm(feat1) * norm(feat2))
    return min(0.5 + 0.5 * cosine_value, 1.0)


def compare_feat(query_feat, gallery_feats):
    simi_list = []
    for i, gallery_feat in enumerate(gallery_feats):
        simi = cosine_distance(query_feat, gallery_feat)
        simi_list.append([simi, i])
    simi_list.sort(key=lambda ele: ele[0], reverse=True)
    return simi_list


temp_queue = [
    # {
    #     'time': UTC seconds,
    #     'feat': (1024) ndarray,
    #     'plate': 'corresponding plate string'
    # }
]


def search_temp_queue(img):
    curr_feat = compute_feat(img)
    return compare_temp_queue(curr_feat)


def compare_temp_queue(curr_feat):
    INTERVAL_THRESH = 60 * 5  # 5min
    SIMI_THRESH = 0.5  # similarity threshold

    # delete outdated nodes
    curr_time = time.time()
    idx, size = 0, len(temp_queue)
    while idx < size and curr_time - temp_queue[idx]['time'] > INTERVAL_THRESH:
        idx += 1
    del temp_queue[:idx]

    max_simi, final_plate = 0.0, ''
    for node in temp_queue:
        simi = cosine_distance(curr_feat, node['feat'])
        if simi > SIMI_THRESH and simi > max_simi:
            max_simi = simi
            final_plate = node['plate']
    return final_plate


def save_temp_queue(img, plate):
    feat = compute_feat(img)
    temp_queue.append({
        'time': time.time(),
        'feat': feat,
        'plate': plate
    })


__all__ = ['compute_feat', 'compare_feat', 'search_temp_queue', 'save_temp_queue']
