#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import random
import numpy as np
from pascal_voc_io import *

def get_line():
    '''
    从标注xml中取出单条字符串子图
    :return:
    '''
    SRC_IMG_DIR = '/home/wz/DataSets/LICENCES/组织机构代码证/IMG'
    SRC_XML_DIR = '/home/wz/DataSets/LICENCES/组织机构代码证/XML'
    LINE_SAVE_DIR = '/home/wz/DataSets/LICENCES/组织机构代码证/LINE_IMG'

    imgs = os.listdir(SRC_IMG_DIR)
    xmls = os.listdir(SRC_XML_DIR)

    # val rect.
    offsetx = 2
    width_multiply = 3
    ver_expand = 10

    line_id = 0
    for xml in xmls:
        xml_path = os.path.join(SRC_XML_DIR, xml)
        img_path = os.path.join(SRC_IMG_DIR, xml[:-4] + '.jpg')
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        reader = PascalVocReader(xml_path)
        shapes = reader.getShapes()
        for shape in shapes:
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            height = left_bottom[1] - left_top[1]
            width = right_bottom[0] - left_top[0]
            val_line = img[left_top[1] - ver_expand: left_bottom[1] + ver_expand,
                       right_top[0] + offsetx: right_top[0] + offsetx + width * width_multiply]
            cv2.imwrite(os.path.join(LINE_SAVE_DIR, str(line_id) + '.jpg'), val_line)
            line_id += 1

        print('Processed file %s.' % xml)


def reshape_pva(ori_img, to_height = 640, to_width = 1056):
    height = ori_img.shape[0]
    width = ori_img.shape[1]

    scale = width * 1.0 / height

    standard_width = to_width
    standard_height = int(standard_width / scale)

    hori_border = True
    if standard_height > to_height:
        scale = height * 1.0 / width
        standard_height = to_height
        standard_width = int(standard_height / scale)
        hori_border = False

    res_img = np.zeros((to_height, to_width, 3), dtype=np.uint8)
    standard_img = cv2.resize(ori_img, dsize=(standard_width, standard_height), interpolation=cv2.INTER_CUBIC)
    margin = 0
    if hori_border:
        margin = int((to_height - standard_height) / 2)
        res_img[margin:margin + standard_height] = standard_img
    else:
        margin = int((to_width - standard_width) / 2)
        res_img[:, margin:margin + standard_width] = standard_img
    zoomout_scale_w = standard_width * 1.0 / width
    zoomout_scale_h = standard_height * 1.0 / height

    return res_img, zoomout_scale_h, zoomout_scale_w, margin, hori_border

def make_stitch_img_xml(SRC_IMG_DIR,SRC_XML_DIR, IMG_SAVE_DIR, XML_SAVE_DIR, NUM_LINES, IMG_HEIGHT = 640,
                        IMG_WIDTH = 1056, NUM_SAMPLES = 1000):
    '''
    stitch three line images into one and save coresponding xml.
    :return:
    '''

    imgs = os.listdir(SRC_IMG_DIR)
    xmls = os.listdir(SRC_XML_DIR)

    for i in range(NUM_SAMPLES):
        xml_name_0 = random.choice(xmls)
        xml_name_1 = random.choice(xmls)
        xml_name_2 = random.choice(xmls)
        img_name_0 = xml_name_0[:-4]+'.jpg'
        img_name_1 = xml_name_1[:-4]+'.jpg'
        img_name_2 = xml_name_2[:-4]+'.jpg'
        xml_0 = os.path.join(SRC_XML_DIR, xml_name_0)
        xml_1 = os.path.join(SRC_XML_DIR, xml_name_1)
        xml_2 = os.path.join(SRC_XML_DIR, xml_name_2)

        img_0 = cv2.imread(os.path.join(SRC_IMG_DIR, img_name_0))
        img_1 = cv2.imread(os.path.join(SRC_IMG_DIR, img_name_1))
        img_2 = cv2.imread(os.path.join(SRC_IMG_DIR, img_name_2))
        if img_0 is None:continue
        if img_1 is None:continue
        if img_2 is None:continue
        img_0, zoom_scale_h_0, zoom_scale_w_0, margin_0, hori_0 = reshape_pva(img_0, int(IMG_HEIGHT / NUM_LINES), IMG_WIDTH)
        img_1, zoom_scale_h_1, zoom_scale_w_1, margin_1, hori_1 = reshape_pva(img_1, int(IMG_HEIGHT / NUM_LINES), IMG_WIDTH)
        img_2, zoom_scale_h_2, zoom_scale_w_2, margin_2, hori_2 = reshape_pva(img_2, int(IMG_HEIGHT / NUM_LINES), IMG_WIDTH)

        pva_img = np.zeros(shape = [IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        pva_img[0:  int(IMG_HEIGHT / NUM_LINES)]  = img_0
        pva_img[int(IMG_HEIGHT / NUM_LINES): 2 * int(IMG_HEIGHT / NUM_LINES)] = img_1
        pva_img[2 * int(IMG_HEIGHT / NUM_LINES): 3 * int(IMG_HEIGHT / NUM_LINES)] = img_2

        cv2.imwrite(os.path.join(IMG_SAVE_DIR, str(i) + '.jpg'), pva_img)
        xml_writer = PascalVocWriter(foldername=IMG_SAVE_DIR, filename=str(i),
                                     imgSize=pva_img.shape)

        sc_w = [zoom_scale_w_0, zoom_scale_w_1, zoom_scale_w_2]
        sc_h = [zoom_scale_h_0, zoom_scale_h_1, zoom_scale_h_2]
        margin = [margin_0, margin_1, margin_2]
        hori = [hori_0, hori_1, hori_2]

        for index, xml in enumerate([xml_0, xml_1, xml_2]):
            xml_reader = PascalVocReader(xml)
            for shape in xml_reader.getShapes():
                points = shape[1]
                left_top = points[0]
                right_top = points[1]
                right_bottom = points[2]
                left_bottom = points[3]
                label = shape[0]
                #if label!='word': continue

                offset = [0, 0]
                if index == 0:
                    offset = [0, 0]
                elif index == 1:
                    offset = [0, int(IMG_HEIGHT / NUM_LINES)]
                elif index == 2:
                    offset = [0, 2 * int(IMG_HEIGHT / NUM_LINES)]

                if hori[index]:
                    xml_writer.addBndBox(int(int(left_top[0]) * sc_w[index] + offset[0]),
                                         int(int(left_top[1]) * sc_h[index] + margin[index] + offset[1]),
                                         int(int(right_bottom[0] * sc_w[index] + offset[0])),
                                         int(int(right_bottom[1] * sc_h[index] + margin[index] + offset[1])),
                                         label)
                else:
                    xml_writer.addBndBox(int(int(left_top[0]) * sc_w[index] + margin[index] + offset[0]),
                                         int(int(left_top[1]) * sc_h[index] + offset[1]),
                                         int(int(right_bottom[0] * sc_w[index] + margin[index] + offset[0])),
                                         int(int(right_bottom[1] * sc_h[index] + offset[1])),
                                         label)
        xml_writer.save(os.path.join(XML_SAVE_DIR, str(i) + '.xml'))
        print('Saved %d images successfully!' % i)


def make_stitch_img(srcroot, dstroot):
    '''
    拼接字符串图像为pvashape, 用作测试
    :param srcroot:
    :param dstroot:
    :return:
    '''
    imgs = os.listdir(srcroot)
    for i in range(1000):
        names = [random.choice(imgs), random.choice(imgs),random.choice(imgs)]
        pathes = [os.path.join(srcroot, xxx) for xxx in names]
        img_0 = cv2.imread(pathes[0])
        img_1 = cv2.imread(pathes[1])
        img_2 = cv2.imread(pathes[2])

        IMG_HEIGHT, IMG_WIDTH, NUM_LINES = 640, 1056, 3
        pva_img = np.zeros(shape=[IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        img_0, zoom_scale_h_0, zoom_scale_w_0, margin_0, hori_0 = reshape_pva(img_0, int(IMG_HEIGHT / NUM_LINES),
                                                                              IMG_WIDTH)
        img_1, zoom_scale_h_1, zoom_scale_w_1, margin_1, hori_1 = reshape_pva(img_1, int(IMG_HEIGHT / NUM_LINES),
                                                                              IMG_WIDTH)
        img_2, zoom_scale_h_2, zoom_scale_w_2, margin_2, hori_2 = reshape_pva(img_2, int(IMG_HEIGHT / NUM_LINES),
                                                                              IMG_WIDTH)
        pva_img[0:  int(IMG_HEIGHT / NUM_LINES)] = img_0
        pva_img[int(IMG_HEIGHT / NUM_LINES): 2 * int(IMG_HEIGHT / NUM_LINES)] = img_1
        pva_img[2 * int(IMG_HEIGHT / NUM_LINES): 3 * int(IMG_HEIGHT / NUM_LINES)] = img_2

        cv2.imwrite(os.path.join(dstroot, str(i)+'.jpg'),pva_img)



if __name__ == '__main__':
    # make_stitch_img("/home/wz/Desktop/sss","/home/wz/Desktop/sss_sticth")
    SRC_IMG_DIR = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages'
    SRC_XML_DIR = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    IMG_SAVE_DIR = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages_pva'
    XML_SAVE_DIR = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations_pva'
    NUM_LINES = 3
    IMG_HEIGHT = 640
    IMG_WIDTH = 1056
    NUM_SAMPLES = 50000
    make_stitch_img_xml(SRC_IMG_DIR, SRC_XML_DIR,
                        IMG_SAVE_DIR, XML_SAVE_DIR,
                        NUM_LINES=NUM_LINES,
                        IMG_HEIGHT=IMG_HEIGHT,
                        IMG_WIDTH=IMG_WIDTH,
                        NUM_SAMPLES=NUM_SAMPLES)