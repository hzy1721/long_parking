import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
# 初始化特征点检测器
kp_detector = cv2.GFTTDetector_create(maxCorners=300,
                                      qualityLevel=0.01,
                                      minDistance=100,
                                      blockSize=3)
prev_frame = None
prev_kps = {}
tracker_dist = {
    # track_id: [dist1, dist2, ...]
}


def optical_flow_LK(curr_frame, tracks):
    global prev_frame, prev_kps
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_kps = {}
    dist_info = {}
    DIST_THRESH = 5
    for x1, y1, x2, y2, track_id in tracks:
        if prev_kps and track_id not in prev_kps:
            continue
        car_gray = curr_gray[y1:y2, x1:x2]
        curr_car_kps = kp_detector.detect(car_gray)
        curr_car_kps = np.array([(kp.pt[0] + x1, kp.pt[1] + y1) for kp in curr_car_kps], dtype='float32').reshape(
            (-1, 1, 2))
        if prev_frame is None or not prev_kps:
            curr_kps[track_id] = curr_car_kps
            continue
        prev_car_kps = prev_kps[track_id]
        curr_car_kps, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_car_kps, curr_car_kps)
        prev_matched_kps = []
        curr_matched_kps = []
        for i, matched in enumerate(status):
            if matched and kp_in_box(curr_car_kps[i], (x1, y1, x2, y2)):
                prev_matched_kps.append(prev_car_kps[i])
                curr_matched_kps.append(curr_car_kps[i])
        prev_matched_kps = np.array(prev_matched_kps)
        curr_matched_kps = np.array(curr_matched_kps)
        curr_kps[track_id] = curr_matched_kps
        dist = cv2.norm(prev_matched_kps, curr_matched_kps)
        if dist < DIST_THRESH:
            dist_info[track_id] = dist
    prev_frame, prev_kps = curr_frame, curr_kps
    return curr_kps, dist_info


def long_parking_warning(dist_info):
    global tracker_dist
    warning_cars = []
    new_tracker_dist = {}
    FRAME_THRESH = 25 * 60 * 2  # 25FPS 2min
    for track_id in dist_info:
        dist = dist_info[track_id]
        if track_id in tracker_dist:
            tracker_dist[track_id].append(dist)
            new_tracker_dist[track_id] = tracker_dist[track_id]
        else:
            new_tracker_dist[track_id] = [dist]
        if len(new_tracker_dist[track_id]) > FRAME_THRESH:
            warning_cars.append([track_id, new_tracker_dist[track_id]])
    tracker_dist = new_tracker_dist
    logger.debug(f'tracker_dist: {tracker_dist}')
    if warning_cars:
        logger.debug(f'warning_cars: {warning_cars}')
    return warning_cars


def kp_in_box(kp, box):
    """
    检查特征点是否在框内
    :param kp: shape (1, 2)
    :param box: [x1, y1, x2, y2]
    :return: bool
    """
    x, y = kp[0][0], kp[0][1]
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
