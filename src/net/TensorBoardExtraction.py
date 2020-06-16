from collections import defaultdict, namedtuple
from typing import List
import argparse
import os
import tensorflow as tf
from tensorflow.core.util import event_pb2
from PIL import Image
import logging


# Set logging
FORMAT = '[%(asctime)s [%(name)s][%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
LOG = logging.getLogger('TensorBoardExtraction')

# set data format
TensorBoardImage = namedtuple("TensorBoardImage", ["topic", "image", "count"])


def extract_images_from_event(event_filename: str, image_tags: List[str]):
    LOG.warning('Extracting log images by given topics...')
    topic_counter = defaultdict(lambda: 0)
    serialized_examples = tf.data.TFRecordDataset(event_filename)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for v in event.summary.value:
            if v.tag in image_tags:  # 'Train/Camera_Depth','Train/Lidar_Depth_Err', 'Train/Lidar_Depth_Fix'

                if v.HasField('image'):  # event for images using tensor field
                    tf_img = tf.image.decode_png(v.image.encoded_image_string)  # [H, W, C]
                    np_img = tf_img.numpy()

                    topic_counter[v.tag] += 1

                    cnt = topic_counter[v.tag]
                    tbi = TensorBoardImage(topic=v.tag, image=np_img, count=cnt)

                    yield tbi


def save_img_to_dir(images: List[TensorBoardImage], dir: str):
    if not os.path.isdir(dir):
        LOG.warning('Targeting directory does not exist! Creating new one...')
        os.mkdir(dir)
    LOG.warning('Target directory prepared.')

    for image in images:
        img = Image.fromarray(image.image)
        img.save(os.path.join(dir, '%s-%d.png')%(image.topic.replace('/','-'), image.count))
    LOG.warning('All images save to the target directory.')
    return


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, help='Path to the TensorBoard log file.', required=True)
    parser.add_argument('--topics', nargs='+', help='Path to the TensorBoard log file.', required=True)
    parser.add_argument('--dir', type=str, help='Path to the saving folder.', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # setup argparser
    args = arg_parser()
    print(args)
    img_list = [data for data in extract_images_from_event(args.log, args.topics)]
    save_img_to_dir(img_list, args.dir)
