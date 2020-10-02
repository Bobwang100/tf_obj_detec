# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.


from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pylab

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
#   raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup

# In[2]:


# This is needed to display the images.
# get_ipython().magic(u'matplotlib inline')


# ## Object detection imports
# Here are the imports from the object detection module.


from utils import label_map_util

from utils import visualization_utils as vis_util
import cv2

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
MODEL_NAME = 'fast_ssd_detction'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'character.pbtxt')

NUM_CLASSES = 10

# ## Download Model


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
'''
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''

# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, pic) for pic in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def cal_single_iou(box1, box2):
    assert box1.shape == (4, ), 'shape error'
    assert box2.shape == (4, ), 'shape error'
    ymin1, xmin1, ymax1, xmax1  = box1
    ymin2, xmin2, ymax2, xmax2  = box2
    xx1 = np.max((xmin1, xmin2))   # 交集框的左上角
    yy1 = np.max((ymin1, ymin2))
    xx2 = np.min((xmax1, xmax2))   # 交集框右下角
    yy2 = np.min((ymax1, ymax2))
    area1 = (xmax1 - xmin1)*(ymax1 - ymin1)
    area2 = (xmax2 - xmin2)*(ymax2 - ymin2)
    inter_area = (np.max((0, xx2-xx1))*(np.max((0, yy2-yy1))))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou


# In[ ]:

with detection_graph.as_default():
    # Get handles to input and output tensors
    sess = tf.Session()
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

df = pd.DataFrame({'file_name': [], 'file_code': []})
num = 0
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image_np, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    # output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # img = cv2.imread('./'+image_path)
    # print(image_path, output_dict['detection_classes'][:10], np.shape(output_dict['detection_classes']))
    # print(image_path, output_dict['detection_scores'][:10], np.shape(output_dict['detection_scores']))
    # print(image_path, output_dict['detection_boxes'], np.shape(output_dict['detection_boxes']))
    filter_cond = output_dict['detection_scores'] > 0.2
    filter_classes = output_dict['detection_classes'][filter_cond]
    filter_classes[filter_classes == 10] = 0
    filter_scores = output_dict['detection_scores'][filter_cond]
    filter_boxes = output_dict['detection_boxes'][filter_cond]
    iou_per_image = []
    # print('filter_box_shape:', image_path, filter_boxes.shape[0])
    all_del_idx = []
    for idx, box in enumerate(filter_boxes):
        for sub_i in range(idx+1, filter_boxes.shape[0]):
            iou = cal_single_iou(box, filter_boxes[sub_i])
            iou_per_image.append(iou)
            if iou > 0.7:
                del_idx = idx if filter_scores[idx] < filter_scores[sub_i] else sub_i
                all_del_idx.append(del_idx)
    #if all_del_idx:
        #print('DELETED:', filter_scores[all_del_idx], filter_classes[all_del_idx], filter_boxes[all_del_idx])
       # print('before:', image_path, filter_classes)
    filter_scores = np.delete(filter_scores, all_del_idx)
    filter_classes = np.delete(filter_classes, all_del_idx)
    filter_boxes = np.delete(filter_boxes, all_del_idx, 0)
    # print('iou_per_image:', image_path, len(iou_per_image), iou_per_image)
    final_classes = [filter_classes[i] for i in np.argsort(filter_boxes[:, 1])]
    final_classes = ''.join(str(i) for i in final_classes)
    # if all_del_idx:
    #     print('after:', image_path, final_classes)
    # h, w = img.shape[:2]
    # for i in range(len(filter_classes)):   # （ymin, xmin, ymax, xmax）
    #     cv2.rectangle(img, (int(filter_boxes[i][1]*w), int(filter_boxes[i][0]*h)),
    #                   (int(filter_boxes[i][3]*w), int(filter_boxes[i][2]*h)), (np.random.randint(0, 255), np.random.randint(0, 255),
    #                                                                  np.random.randint(0, 255)), 1)
    # print('FINAL_RESULT:', final_classes)
    df.loc[num] = [image_path.split('/')[-1], final_classes]
    num += 1
    if num % 100 == 0:
        print(num, '--------->', len(TEST_IMAGE_PATHS))
    # if num > 500:
    # #     break
    # cv2.imshow('img', img)
    # cv2.waitKey(1500)
    # cv2.destroyAllWindows()
df = df.sort_values('file_name')
df.to_csv('./submit0623.csv', index=None)

# plt.figure(figsize=IMAGE_SIZE)
# plt.imshow(image_np)
# pylab.show()
