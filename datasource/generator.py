import os
import logging
import cv2

logger = logging.getLogger(__name__)


def channel_generator(channel):
    root = '/data1/interface/shangxiantest/161'
    frame_dir = os.path.join(root, channel)
    # delete overdue images
    img_files = sorted(os.listdir(frame_dir))
    for filename in img_files:
        img_path = os.path.join(frame_dir, filename)
        os.remove(img_path)
    logger.info(f'Delete {len(img_files)} images')
    run_interval = 25 * 10  # 10s
    while True:
        img_files = sorted(os.listdir(frame_dir))
        if len(img_files) < run_interval:
            continue
        img_files = img_files[:run_interval]
        for filename in img_files:
            logger.debug(f'Channel image: {filename}')
            img_path = os.path.join(frame_dir, filename)
            img = cv2.imread(img_path)
            yield img
            os.remove(img_path)


def video_generator(video):
    cap = cv2.VideoCapture(video)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        yield img


def frame_dir_generator(frame_dir):
    for root, dirs, files in os.walk(frame_dir):
        for filename in sorted(files):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            yield img
