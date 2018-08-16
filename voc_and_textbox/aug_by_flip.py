#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function, division
import sys
import os
import cv2

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

import scipy
pi     = scipy.pi
dot    = scipy.dot
sin    = scipy.sin
cos    = scipy.cos
ar     = scipy.array
rand   = scipy.rand
arange = scipy.arange
rad    = lambda ang: ang*pi/180
######################################################################
# 用于增强检测的数据
# 结果就存在原路径对应文件夹下，会自动将原样本拷贝进去。
######################################################################

def flip_shapes(imgshape, shapes, mode='leftrt'):#'updown'
    w = imgshape[1]
    h = imgshape[0]
    new_shapes = []
    for shape in shapes:
        points = shape[1]
        left_top = points[0]
        right_top = points[1]
        right_bottom = points[2]
        left_bottom = points[3]
        old_pts = [left_top, right_bottom, right_top, left_bottom]
        new_pts = []
        if mode=='leftrt':
            for pt in old_pts:
                new_pts.append([w - pt[0], pt[1]])
        else:
            for pt in old_pts:
                new_pts.append([pt[0], h - pt[1]])
        new_shapes.append(new_pts)
    return new_shapes

def rotate2D(pts, ang=pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return dot(pts, ar([[cos(ang),sin(ang)],[-sin(ang),cos(ang)]]))

def rotate_shapes(imgshape, shapes, theta = pi):#pi, pi / 2, pi * 3 / 2
    w = imgshape[1]
    h = imgshape[0]
    offsetx = w / 2
    offsety = h / 2
    offset = (offsetx, offsety)

    #m = cv2.getRotationMatrix2D(offset, theta, 1)
    new_shapes = []
    for shape in shapes:
        points = shape[1]
        left_top = points[0]
        right_top = points[1]
        right_bottom = points[2]
        left_bottom = points[3]
        old_pts_centered = np.array([[left_top[0]-offsetx, left_top[1]-offsety],
                                    [right_bottom[0]-offsetx,right_bottom[1]-offsety],
                                    [right_top[0]-offsetx,right_top[1]-offsety],
                                    [left_bottom[0]-offsetx,left_bottom[1]-offsety]])
        # old_pts = np.array([left_top, right_bottom, right_top, left_bottom])
        #先以中心为原点旋转９０或者２７０然后加上置换后的横纵坐标
        if abs(abs(theta) - pi) < 0.01:
            new_pts = rotate2D(old_pts_centered, theta) + [offsetx, offsety]
        else:
            new_pts = rotate2D(old_pts_centered, theta) + [offsety, offsetx]
        new_shapes.append(new_pts)
    return new_shapes

def augment(xml_name):
    basename = xml_name[0:-4]
    img_name = basename + '.jpg'
    if not os.path.exists(os.path.join(img_dir, img_name)):
        return

    img = cv2.imread(os.path.join(img_dir, img_name))
    # img_updown = cv2.flip(img, 0)
    # img_leftrt = cv2.flip(img, 1)
    # img_updown_leftrt = cv2.flip(img_updown, 1)
    # img_180 = cv2.flip(img, -1)
    # cv2.imwrite(os.path.join(res_img_dir, basename+'.jpg'),img)
    # cv2.imwrite(os.path.join(res_img_dir, basename+'-leftrt.jpg'), img_leftrt)
    # cv2.imwrite(os.path.join(res_img_dir, basename+'-updown.jpg'), img_updown)
    reader = PascalVocReader(os.path.join(xml_dir, xml_name))
    # ubuntu文件夹中显示是正的但是property中宽和高是反过来的，labelme中读进去宽高是正常的但是显示和文件夹不一致
    # 这个地方的cv读进来是按照文件夹显示中读取的，有点绕
    if reader.height != img.shape[0]:
        print(xml_name , ' is with invalid height and width.')
        shutil.move(os.path.join(xml_dir,xml_name),os.path.join(xml_dir,basename+'-invalid.xml'))
        return

    shapes = reader.getShapes()
    # writer_lwftrt = PascalVocWriter('VOC', basename + '-leftrt.xml', img_leftrt.shape)
    # leftrt_shapes = flip_shapes(img.shape, shapes)
    # for poly, shape in zip(leftrt_shapes, reader.getShapes()):
    #     np_poly = np.array(poly)
    #     writer_lwftrt.addBndBox(min(np_poly[:, 0]), min(np_poly[:, 1]), max(np_poly[:, 0]), max(np_poly[:, 1]), shape[0])
    # writer_lwftrt.save(os.path.join(res_xml_dir, basename + '-leftrt.xml'))
    #
    # writer_updown = PascalVocWriter('VOC', basename + '-updown.xml', img_leftrt.shape)
    # updown_shapes = flip_shapes(img.shape, shapes, 'updown')
    # for poly, shape in zip(updown_shapes, reader.getShapes()):
    #     np_poly = np.array(poly)
    #     writer_updown.addBndBox(min(np_poly[:, 0]), min(np_poly[:, 1]), max(np_poly[:, 0]), max(np_poly[:, 1]), shape[0])
    # writer_updown.save(os.path.join(res_xml_dir, basename + '-updown.xml'))

    # img_90 = np.rot90(img, 1)
    img_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    writer_90 = PascalVocWriter('VOC', basename + '-90.xml', img_90.shape)
    shapes_90 = rotate_shapes(img.shape, shapes, -pi / 2)
    for poly, shape in zip(shapes_90, reader.getShapes()):
        np_poly = np.array(poly)
        writer_90.addBndBox(int(min(np_poly[:, 0])), int(min(np_poly[:, 1])), int(max(np_poly[:, 0])),
                             int(max(np_poly[:, 1])), shape[0])
    writer_90.save(os.path.join(res_xml_dir, basename + '-90.xml'))


    img_180 = cv2.rotate(img, cv2.ROTATE_180)
    writer_180 = PascalVocWriter('VOC', basename + '-180.xml', img_180.shape)
    shapes_180 = rotate_shapes(img.shape, shapes, -pi )
    for poly, shape in zip(shapes_180, reader.getShapes()):
        np_poly = np.array(poly)
        writer_180.addBndBox(int(min(np_poly[:, 0])), int(min(np_poly[:, 1])), int(max(np_poly[:, 0])),
                             int(max(np_poly[:, 1])), shape[0])
    writer_180.save(os.path.join(res_xml_dir, basename + '-180.xml'))

    img_270 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    writer_270 = PascalVocWriter('VOC', basename + '-270.xml', img_270.shape)
    shapes_270 = rotate_shapes(img.shape, shapes, -pi * 3 /2)
    for poly, shape in zip(shapes_270, reader.getShapes()):
        np_poly = np.array(poly)
        writer_270.addBndBox(int(min(np_poly[:, 0])), int(min(np_poly[:, 1])), int(max(np_poly[:, 0])), int(max(np_poly[:, 1])), shape[0])
    writer_270.save(os.path.join(res_xml_dir, basename + '-270.xml'))

    shutil.copy(os.path.join(xml_dir, basename+'.xml'),os.path.join(res_xml_dir, basename+'.xml'))
    cv2.imwrite(os.path.join(res_img_dir, basename+'.jpg'), img)
    cv2.imwrite(os.path.join(res_img_dir, basename+'-90.jpg'), img_90)
    cv2.imwrite(os.path.join(res_img_dir, basename+'-180.jpg'), img_180)
    cv2.imwrite(os.path.join(res_img_dir, basename+'-270.jpg'), img_270)


if __name__ == '__main__':
    augment_times = 3
    voc_dir = "/home/wz/DataSets/dataforocr-zsm/sfz/front_aug/IDCard-ROI/"
    xml_dir = os.path.join(voc_dir, 'XML')
    img_dir = os.path.join(voc_dir, 'IMG')
    res_xml_dir = os.path.join(voc_dir, 'Annotations')
    res_img_dir = os.path.join(voc_dir, 'JPEGImages')

    xmls = os.listdir(xml_dir)
    for xml in xmls:
        augment(xml)
    # pool = mp.Pool(processes=None)
    # pool.map(augment, xmls)
    # filter_invalid_xml(res_xml_dir, res_img_dir, '/tmp/xml_filter')