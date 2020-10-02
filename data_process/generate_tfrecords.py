import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# os.chdir('/home/zzf/tensorflow/models/research/object_detection')

flags = tf.app.flags
flags.DEFINE_string('csv_input', './char_train.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', './char_train.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'one':     # 需改动
        return 1
    if row_label == 'two':     # 需改动
        return 2
    if row_label == 'three':     # 需改动
        return 3
    if row_label == 'four':     # 需改动
        return 4
    if row_label == 'five':     # 需改动
        return 5
    if row_label == 'six':     # 需改动
        return 6
    if row_label == 'seven':     # 需改动
        return 7
    if row_label == 'eight':     # 需改动
        return 8
    if row_label == 'nine':     # 需改动
        return 9
    if row_label == 'zero':     # 需改动
        return 10


    # else:
    #     None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(min(row['xmax'] / width, 1.0))
        ymins.append(row['ymin'] / height)
        ymaxs.append(min(row['ymax'] / height, 1.0))
        if row['xmax'] / width > 1 or row['ymax'] / height > 1:
            print('!!!ATTENTION', filename, row['xmax'] / width, row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))    # .encode('utf8')
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # path = os.path.join(os.getcwd(), '../images/train/')         #  需改动
    path = '../images/train/'         #  需改动
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    group_id = 0
    for group in grouped:
        tf_example = create_tf_example(group, path)
        group_id += 1
        if group_id % 100 == 0:
            print('finish ', group_id, '---->', len(grouped))
        # if group_id > 100:
        #     break
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()