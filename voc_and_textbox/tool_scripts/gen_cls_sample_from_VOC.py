#!/usr/bin/env python
#-- coding:utf-8 --
import os.path as osp
import os
import cv2
import random
import numpy as np
from pascal_voc_io import  *


cur_dir = osp.dirname(__file__)

#size for classification net's input.
RESIZE_WIDTH = 96
RESIZE_HEIGHT = 96
IMAGE_SIZE = (RESIZE_WIDTH, RESIZE_HEIGHT)

#pathes of input xml and image, and output image path.
xml_folder        = '/home/wz/DataSets/dataforocr-zsm/sfz/front_aug/IDCard-ROI/XML'
image_src_folder  = '/home/wz/DataSets/dataforocr-zsm/sfz/front_aug/IDCard-ROI/IMG'
image_save_root   = '/home/wz/DataSets/dataforocr-zsm/sfz/front_aug/IDCard-ROI/ROI-CROP'
classes_name_txt  = '/home/wz/Documents/idcard-name.txt'

if not os.path.exists(image_save_root):
    os.mkdir(image_save_root)

CLASSES = []
fin = open(classes_name_txt)
for line in fin.readlines(  ):
    CLASSES.append(line[:-1])
    if not os.path.isdir(os.path.join(image_save_root, line[:-1])):
        os.mkdir(os.path.join(image_save_root, line[:-1]))

class_index = dict(zip(CLASSES, len(CLASSES)*[0]))
print class_index

for xml_name in os.listdir(xml_folder):
    basename = xml_name[:-4]
    xml_reader = PascalVocReader(os.path.join(xml_folder, xml_name))
    if len(xml_reader.getShapes()) > 0:
        img = cv2.imread(os.path.join(image_src_folder, basename + '.jpg'))
        if img is None: continue
        #ubuntu文件夹中显示是正的但是property中宽和高是反过来的，labelme中读进去是正常的，这个地方的cv读进来是按照文件夹显示中读取的，有点绕
        if xml_reader.height != img.shape[0]: continue

        for shape in xml_reader.getShapes():
            label = shape[0]
            class_index[label] += 1
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            crop_w = np.floor((right_top[1] - left_top[1]) / 8)
            crop_h = np.floor((right_top[0] - right_bottom[0]) / 8)
            offsetx = 0#random.randint(0, crop_w)
            offsety = 0#random.randint(0, crop_h)
            offsetw = 0#random.randint(0, crop_w)
            offseth = 0#random.randint(0, crop_h)

            xmin = min(left_top[0],right_top[0],right_bottom[0],left_bottom[0])
            xmax = max(left_top[0],right_top[0],right_bottom[0],left_bottom[0])
            ymin = min(left_top[1],right_top[1],right_bottom[1],left_bottom[1])
            ymax = max(left_top[1],right_top[1],right_bottom[1],left_bottom[1])
            print(xmin, xmax, ymin, ymax)

            # crop_img = img[max(0, left_top[1]+offsety): min(img.shape[0]-offseth-offsety, left_bottom[1]),
            #            max(0,left_top[0]+offsetx): min(img.shape[1]-offsetw-offsetx,right_top[0])]
            crop_img = img[ymin:ymax, xmin:xmax]
            if crop_img is None:
                continue
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0: continue
            # resized_crop_img = cv2.resize(crop_img, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(image_save_root, label, label + '_' + str(class_index[label]) + ".jpg"), crop_img)
    print('Processed xml %s.' % xml_name)