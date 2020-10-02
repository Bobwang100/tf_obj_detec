import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
# os.chdir('/home/zzf/tensorflow/models/research/object_detection/images/test')
path = './val.txt'

def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    line format: line_index, img_path, img_width, img_height, [box_info_1 (5 number)], ...
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
        img_width: int.
        img_height: int
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    assert len(
        s) > 8, 'Annotation error! Please check your annotation file. Make sure there is at least one target object ' \
                'in each image. '
    line_idx = int(s[0])
    pic_path = s[1]
    img_width = int(s[2])
    img_height = int(s[3])
    s = s[4:]
    assert len(
        s) % 5 == 0, 'Annotation error! Please check your annotation file. Maybe partially missing some coordinates?'
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels, img_width, img_height


def get_image_and_label():
    image_label_dict = {}
    for line in open('./val.txt').readlines():
        line_idx, pic_path, boxes, labels, img_width, img_height = parse_line(line)
        # print(line_idx, pic_path, boxes, labels, img_width, img_height)
        # image_name = pic_path.strip().split('/')[-1]
        image_label_dict[line_idx] = [pic_path, img_width, img_height, labels, boxes]
    return image_label_dict

int_char_dict = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
                 }

def txt_to_csv(img_label):
    info_list = []
    for idx in range(len(img_label)):
        for num in range(len(img_label[idx][3])):
            value = (img_label[idx][0].split('/')[-1],
                     img_label[idx][1],
                     img_label[idx][2],
                     int_char_dict[img_label[idx][3][num]],
                     img_label[idx][4][num][0],
                     img_label[idx][4][num][1],
                     img_label[idx][4][num][2],
                     img_label[idx][4][num][3],
            )
            info_list.append(value)
    # xml_list = []
    #
    # for xml_file in glob.glob(path + '/*.xml'):
    #     tree = ET.parse(xml_file)
    #     root = tree.getroot()
    #     for member in root.findall('object'):
    #         value = (root.find('filename').text,
    #                  int(root.find('size')[0].text),
    #                  int(root.find('size')[1].text),
    #                  member[0].text,
    #                  int(member[4][0].text),
    #                  int(member[4][1].text),
    #                  int(member[4][2].text),
    #                  int(member[4][3].text)
    #                  )
    #         xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(info_list, columns=column_name)
    return xml_df


def main():
    img_label_dict = get_image_and_label()
    txt_df = txt_to_csv(img_label_dict)
    txt_df.to_csv('char_val.csv', index=None)
    print('Successfully converted xml to csv.')


main()
