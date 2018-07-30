#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sys
import os

sys.path.append('../')

import numpy as np
from scipy import ndimage
from scipy import misc

from imgaug import augmenters as iaa
import imgaug as ia
import shutil

from voc_and_textbox.tool_scripts.pascal_voc_io import *
from voc_and_textbox.tool_scripts.filter_empty_and_invalid_box import *

import multiprocessing as mp
######################################################################
# 用于增强检测的数据
# 结果就存在原路径对应文件夹下，会自动将原样本拷贝进去。
######################################################################


# 配置增强参数
seq = iaa.Sequential([
    iaa.Crop(px=(0, 100)),  # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    # iaa.Flipud(0.5),
    iaa.AdditiveGaussianNoise(scale=0.1 * 100),
    iaa.GaussianBlur(sigma=(0.8, 1.5)),
    iaa.Affine(rotate=(-2, 2), mode ='edge'),#mode ='edge'代表以边缘像素填充放射变换后的空洞背景
    # iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    #            # scale images to 80-120% of their size, individually per axis
    #            translate_px={"x": (-10, 10), "y": (-10, 10)}, mode ='edge'),
    iaa.Affine(scale=(0.95, 1.0), mode ='edge'),
    # iaa.Affine(shear=(-20, 20), mode ='edge'),
    iaa.Multiply((0.9, 1.2)),  # change brightness of images (50-150% of original value)
    # iaa.ContrastNormalization((0.8, 1.5))
])

def augment(xml_name):
    img_name = xml_name[0:-4] + '.jpg'
    if not os.path.exists(os.path.join(img_dir, img_name)):
        return

    image = ndimage.imread(os.path.join(img_dir, img_name))
    reader = PascalVocReader(os.path.join(xml_dir, xml_name))

    shapes = reader.getShapes()
    polygons = [shape[1] for shape in shapes]
    keypoints = []
    for polygon in polygons:
        keypoints.extend([ia.Keypoint(point[0], point[1]) for point in polygon])
    keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

    for i in range(augment_times):
        cur_xml_name = xml_name[0: -4] + '_' + str(i)
        writer = PascalVocWriter(foldername='VOC', filename=cur_xml_name, imgSize=image.shape)

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_image(image)
        keypoints_aug = seq_det.augment_keypoints(keypoints)[0]
        # convert from keypoints to array, or else can't make it to list in below loop
        new_points = keypoints_aug.get_coords_array()
        new_polygons = []
        bndBoxs = []
        num = 0
        for polygon in polygons:
            new_polygon = [(new_points[num + ind][0], new_points[num + ind][1]) for ind in range(len(polygon))]
            num = num + len(polygon)
            new_polygons.append(new_polygon)

        for poly, shape in zip(new_polygons, reader.getShapes()):
            np_poly = np.array(poly)
            writer.addBndBox(min(np_poly[:, 0]), min(np_poly[:, 1]), max(np_poly[:, 0]), max(np_poly[:, 1]), shape[0])

        writer.save(os.path.join(res_xml_dir, cur_xml_name + XML_EXT))
        misc.imsave(os.path.join(res_img_dir, cur_xml_name + '.jpg'), image_aug)
    shutil.copy(os.path.join(xml_dir, xml_name), os.path.join(res_xml_dir, xml_name))
    shutil.copy(os.path.join(img_dir, img_name), os.path.join(res_img_dir, img_name))

if __name__ == '__main__':
    augment_times = 5
    voc_dir = "/home/wz/DataSets/LICENCES/ZZJGDMZ/VOC2007/"
    xml_dir = os.path.join(voc_dir, 'XML')
    img_dir = os.path.join(voc_dir, 'IMG')
    res_xml_dir = os.path.join(voc_dir, 'Annotations')
    res_img_dir = os.path.join(voc_dir, 'JPEGImages')

    xmls = os.listdir(xml_dir)
    pool = mp.Pool(processes=None)
    pool.map(augment, xmls)
    filter_invalid_xml(res_xml_dir, res_img_dir, '/tmp/xml_filter')