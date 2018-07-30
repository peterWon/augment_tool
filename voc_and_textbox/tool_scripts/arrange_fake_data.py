# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import multiprocessing as mp
import shutil
from pascal_voc_io import *

#/home/wz/DeepLearning/pytorch_dir/pytorch-CycleGAN-and-pix2pix-master/results/lincese_pix2pix/test_latest/images/00000000_fake_B.png
def single_process(xml):
    base_name = str(400000 + int(xml[:-4]))
    reader = PascalVocReader(os.path.join(xml_dir, xml))
    ori_w, ori_h = reader.width, reader.height

    img_read_name = base_name + '_fake_B.png'
    img_save_name = base_name + '.jpg'
    img_path = os.path.join(fake_img_dir, img_read_name)
    if not os.path.exists(img_path): return

    # 过滤由于黑色边界二值化后出现的纯白图
    img_real_A = base_name + '_real_A.png'
    img_A = cv2.imread(os.path.join(fake_img_dir, img_real_A), cv2.IMREAD_COLOR)

    if img_A is None: return
    img_A_reshape = cv2.resize(img_A, (ori_w, ori_h), cv2.INTER_CUBIC)
    num_empty = 0
    for shape in reader.getShapes():
        lefttop, rttop, rtbt, leftbt = shape[1]
        subimg = img_A_reshape[lefttop[1]:leftbt[1], lefttop[0]:rttop[0], 0]
        if (np.sum(subimg)
            / (255 * subimg.shape[0] * subimg.shape[1])) > 0.95:
            num_empty+=1
    if num_empty >= 3: return

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return

    img_reshape = cv2.resize(img, (ori_w, ori_h), cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(img_save_dir, img_save_name), img_reshape)
    # shutil.copy(os.path.join(xml_dir, xml), os.path.join(xml_save_dir, xml))

    print('Write image %s succeessfuly!' % img_save_name)


def merge_fake_real():
    real_xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations_ori'
    fake_xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations_fake'
    real_img_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages_ori'
    fake_img_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages_fake'
    merge_xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    merge_img_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages'
    for xml in os.listdir(real_xml_dir):
        shutil.copy(os.path.join(real_xml_dir, xml), os.path.join(merge_xml_dir, xml[:-4]+'_real.xml'))
    for img in os.listdir(real_img_dir):
        shutil.copy(os.path.join(real_img_dir, img), os.path.join(merge_img_dir, img[:-4]+'_real.jpg'))

    # for xml in os.listdir(fake_xml_dir):
    #     if int(xml[:-4]) > 3000: continue
    #     shutil.copy(os.path.join(fake_xml_dir, xml), os.path.join(merge_xml_dir, xml[:-4]+'_fake.xml'))
    #     shutil.copy(os.path.join(fake_img_dir, xml[:-4]+'.jpg'), os.path.join(merge_img_dir, xml[:-4] + '_fake.jpg'))



if __name__ == '__main__':
    # xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    # fake_img_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/pix2pix/lincese_pix2pix/test_latest/images'
    # img_save_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages'
    # xml_save_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    #
    # xmls = os.listdir(xml_dir)
    # pool = mp.Pool(None)
    # pool.map(single_process, xmls)

    #debug
    # for xml in xmls:
    #     single_process(xml)

    #merge dataset
    # merge_fake_real()

    xmldir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    xmls = os.listdir(xmldir)
    for xml in xmls:
        basename = xml[:-4]
        bin_xml_name = str(int(basename)+200000)
        fake_xml_name = str(int(basename)+400000)

        shutil.copy(os.path.join(xmldir, xml), os.path.join(xmldir, bin_xml_name+'.xml'))
        shutil.copy(os.path.join(xmldir, xml), os.path.join(xmldir, fake_xml_name+'.xml'))
        print(basename, bin_xml_name, fake_xml_name)