#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sys
import os
sys.path.append('../')

import imgaug as ia
from imgaug import augmenters as iaa
from scipy import ndimage, misc
from tool_scripts.textbox_io import *

import multiprocessing as mp
import cv2 #just for debug


aug_mult = 5

seq = iaa.Sequential([
            iaa.Crop(px=(0, 10)),  # crop images from each side by 0 to 16px (randomly chosen)
            # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            # iaa.Flipud(0.5),
            iaa.Affine(rotate=(-5, 5), mode='edge'),#仿she增强没问题，是在画图的时候最小外接举行已经不再是4,5了，而是得重新计算
            # iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                       # scale images to 80-120% of their size, individually per axis
                      # translate_px={"x": (-8, 8), "y": (-8, 8)}),
            iaa.Affine(scale=(0.6, 1.0), mode='edge'),
            #     shear=(-30, 30)),
            iaa.Multiply((0.7, 1.2)),  # change brightness of images (50-150% of original value)
            iaa.ContrastNormalization((0.9, 1.1))
        ])


def augment(xml_):
    base_name = xml_[0:-4]
    img_name = base_name +'.jpeg'
    if not os.path.exists(os.path.join(img_dir, img_name)):
        return
    image = ndimage.imread(os.path.join(img_dir, img_name))

    reader = TextboxReader(os.path.join(xml_dir, xml_))
    writer = TextboxWriter(img_dir,  xml_, image.shape)

    polygons = [[obj[1],obj[2],obj[3],obj[4]] for obj in reader.get_objects()]

    keypoints = []
    for polygon in polygons:
        keypoints.extend([ia.Keypoint(point[0], point[1]) for point in polygon])
    keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

    for i in range(aug_mult):
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

        writer.set_filename(base_name+'_'+str(i))
        for poly in new_polygons:
            poly = np.array(poly)
            xmin = min(poly[:,0])
            ymin = min(poly[:,1])
            xmax = max(poly[:,0])
            ymax = max(poly[:,1])
            writer.add_bndbox(poly[0,0],poly[0,1],poly[1,0],poly[1,1],
                              poly[2,0],poly[2,1],poly[3,0],poly[3,1],
                              xmin,ymin,xmax,ymax,name='IDcard')

        #Just for debug
        # for poly in new_polygons:
        #     poly = np.array(poly)
        #     xmin = min(poly[:,0])
        #     ymin = min(poly[:,1])
        #     xmax = max(poly[:,0])
        #     ymax = max(poly[:,1])
        #     cv2.rectangle(image_aug, (xmin, ymin), (xmax, ymax), (0,255,0))
        #     cv2.imshow('imgaug', image_aug)
        #     cv2.waitKey(0)

        writer.save(os.path.join(res_xml_dir, base_name + '_' + str(i) + '.xml'))
        writer.clear_bndbox()
        misc.imsave(os.path.join(res_img_dir, base_name + '_' + str(i) + '.jpeg'), image_aug)

    print('augment img %s succesfully!' % xml_)


if __name__ == '__main__':
    xml_dir = '/home/wz/DataSets/dataforocr/sfz/front_tbox_xml'
    img_dir = '/home/wz/DataSets/dataforocr/sfz/front'
    res_xml_dir = '/home/wz/DataSets/dataforocr/sfz/front_aug/xml'
    res_img_dir = '/home/wz/DataSets/dataforocr/sfz/front_aug/img'

    xmls = os.listdir(xml_dir)
    pool = mp.Pool(processes = 4)
    pool.map(augment, xmls)
    print('Done!')