# -*- coding:utf-8 -*-
import json
import os
import cv2
import numpy as np
import math
import lmdb
import sys
import traceback
import multiprocessing as mp
import shutil

from textbox_io import *

DEBUG_MODE = False

def get_y_on_line(x, pt1, pt2):
    if abs(pt1[0] - pt2[0]) < 0.000001 or abs(pt1[1] - pt2[1]) < 0.000001: return pt1[1]
    return int((pt1[1] - pt2[1]) * 1.0 * (x - pt2[0]) / (pt1[0]-pt2[0])  + pt2[1])

def get_x_on_line(y, pt1, pt2):
    if abs(pt1[0] - pt2[0]) < 0.000001 or abs(pt1[1] - pt2[1]) < 0.000001: return pt1[0]
    return int((pt1[0] - pt2[0]) * 1.0 * (y - pt2[1]) / (pt1[1]-pt2[1])  + pt2[0])

def get_pts_on_line(pt1, pt2):
    xbegin, ybegin = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    xend, yend = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    res = []
    if xend - xbegin > yend - ybegin:
        for x in range(xbegin, xend + 1):
            y = get_y_on_line(x, pt1, pt2)
            res.append((x, y))
    else:
        for y in range(ybegin, yend + 1):
            x = get_x_on_line(y, pt1, pt2)
            res.append((x, y))
    return res

def single_xml_processs(img_name):
    imageA = cv2.imread(os.path.join(imgA_dir, img_name), cv2.IMREAD_COLOR)
    xml_path = os.path.join(xml_root_dir, img_name[:-4] + '.xml')
    if imageA is None or not os.path.exists(xml_path):
        return

    reader = TextboxReader(xml_path)
    imageB = np.zeros(shape=(imageA.shape[0],imageA.shape[1]), dtype=np.uint8)
    for poly in reader.get_objects():
        x1, y1, x2, y2, x3, y3, x4, y4 = int(poly[1][0]), int(poly[1][1]), int(poly[2][0]), int(
            poly[2][1]), \
                                         int(poly[3][0]), int(poly[3][1]), int(poly[4][0]), int(
            poly[4][1])

        #trace polyline
        corners = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # for debug
        # cv2.line(imageA,(x1,y1),(x2,y2),(0, 255, 0))
        # cv2.line(imageA,(x3,y3),(x2,y2),(0, 255, 0))
        # cv2.line(imageA,(x1,y1),(x4,y4),(0, 255, 0))
        # cv2.line(imageA,(x4,y4),(x3,y3),(0, 255, 0))
        # cv2.imshow('mmm', imageA)
        # cv2.waitKey()

        linepts = []
        linepts.append(get_pts_on_line(corners[0], corners[1]))
        linepts.append(get_pts_on_line(corners[1], corners[2]))
        linepts.append(get_pts_on_line(corners[2], corners[3]))
        linepts.append(get_pts_on_line(corners[3], corners[0]))

        for pts in linepts:
            for pt in pts:
                imageB[pt[1], pt[0]] = 255
    # cv2.imshow('a', imageA)
    # cv2.imshow('b', imageB)
    # cv2.waitKey()

    cv2.imwrite(os.path.join(imgB_save_dir, img_name), imageB)
    print('Generate couple image of %s succeesfully!' % img_name)


def split_trainval(img_src_dir,  save_root_dir, train = 0.7, test = 0.15, val = 0.15):
    imgsets = {'train': train,'test' : test, 'val':val}
    imgs = os.listdir(img_src_dir)
    pos = 0
    for s in imgsets.keys():
        if not os.path.exists(os.path.join(save_root_dir, s)):
            os.mkdir(os.path.join(save_root_dir, s))
        for i in range(pos, int(pos+len(imgs)*imgsets[s])):
            shutil.copy(os.path.join(img_src_dir,imgs[i]), os.path.join(save_root_dir, s, imgs[i]))
        pos = int(pos+len(imgs)*imgsets[s])

def binarize(img):
    if img is None: return None
    _, binary= cv2.threshold(img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return binary


def make_polyline_data():
    '''
    制作边界线样本，如身份证，行驶证等通过labelme勾选边界后转换到xml的格式样本
    一张输入图生成一张同样大小的边界二值输出图（0：非边界 , 255： 边界）
    :return:
    '''
    imga_dir = '/home/wz/DataSets/CarBrand/border_line/aug/img'
    xml_root_dir = '/home/wz/DataSets/CarBrand/border_line/aug/xml'
    imgb_save_dir = '/home/wz/DeepLearning/pytorch_dir/pytorch-CycleGAN-and-pix2pix-master/datasets/carbrand/B_tmp'
    imgnames = os.listdir(imga_dir)
    # l = len(imgnames)
    # pool = mp.Pool(processes=None)
    # # pool.map(sinle_processs_xml,imgnames, [imga_dir]*l, [xml_root_dir]*l, [imgb_save_dir]*l)
    # res = [pool.apply_async(single_xml_processs, args=(imgname, imga_dir, xml_root_dir, imgb_save_dir))
    #        for imgname in os.listdir(imga_dir)]
    # pool.join()
    # pool.close()

    # for imgname in imgnames:
    #     single_xml_processs(imgname, imga_dir, xml_root_dir,imgb_save_dir)
    split_trainval('/home/wz/DataSets/CarBrand/border_line/aug/img',
                   '/home/wz/DeepLearning/pytorch_dir/pytorch-CycleGAN-and-pix2pix-master/datasets/carbrand/A')

def process(name):
    img = cv2.imread(os.path.join(img_src_dir, name), cv2.IMREAD_GRAYSCALE)
    binary = binarize(img)
    if binary is None: return
    if int(name[:-4]) < 200000:return
    save_name = str(200000 + int(name[:-4]))+'.jpg'
    cv2.imwrite(os.path.join(binary_save_dir, save_name), binary)

if __name__ == '__main__':
    # split_trainval('/home/wz/DataSets/SYNTH_LINE/VOC2007/pix2pix/Binary',
    #               '/home/wz/DataSets/SYNTH_LINE/VOC2007/pix2pix/A', 0.01, 0.98, 0.01)


    #make polyline
    # imgA_dir = '/home/wz/DataSets/LICENCES/invoice_FaPiao/pix2pix/aug_img'
    # xml_root_dir = '/home/wz/DataSets/LICENCES/invoice_FaPiao/pix2pix/aug_xml'
    # imgB_save_dir = '/home/wz/DataSets/LICENCES/invoice_FaPiao/pix2pix/aug_img_b'
    # imgnames = os.listdir(imgA_dir)
    # l = len(imgnames)
    # pool = mp.Pool(processes=None)
    # pool.map(single_xml_processs, imgnames)
