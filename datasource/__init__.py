import logging
from datetime import datetime
import os

from .generator import channel_generator, video_generator, frame_dir_generator

logger = logging.getLogger(__name__)


def get_img_generator(channel, video, frame_dir):
    if channel:
        logger.info(f'数据源(channel): {channel}')
        return channel_generator(channel)
    if video:
        logger.info(f'数据源(视频文件): {video}')
        return video_generator(video)
    if frame_dir:
        logger.info(f'数据源(帧目录): {frame_dir}')
        return frame_dir_generator(frame_dir)
    logger.error('Datasource not provided')
    raise 'Datasource not provided'


def create_output_dir(channel, video, frame_dir):
    time_str = datetime.now().strftime('%Y%m%d%H%M%S')
    if channel:
        out_dir = os.path.join('outputs', f'channel_{channel}_{time_str}')
    elif video:
        out_dir = os.path.join('outputs', f'video_{video.split("/")[-1]}_{time_str}')
    elif frame_dir:
        out_dir = os.path.join('outputs', f'frame_dir_{frame_dir.split("/")[-1]}_{time_str}')
    else:
        raise 'Datasource not provided'
    logger.info(f'输出视频帧的存储位置: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

