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

sys.path.insert(0, '/home/wz/DeepLearning/caffe_dir/easy-pvanet/caffe-fast-rcnn/python/')
import caffe


from textbox_io import *

#--------------------------------------------------------------#
# 根据textbox格式的xml生成相应的feature-map, 并写入caffe的lmdb格式
# 每个关键点占据一个通道,该通道内可以有多个极值,分属每个不同的object.
#--------------------------------------------------------------#

# 是否中心化和归一化
PROCESS_IMG = True
DEBUG_MODE = False

LOCK = mp.Lock()
COUNTER = mp.Value('i', 0)
STOP_TOKEN = 'kill'

def parse_from_json(img_root_dir, json_root_dir, lmdb_path):
    '''
    从labelme的标注格式中解析并生成关键点的热力图并写入lmdb中
    :param img_root_dir:   图像路径
    :param json_root_dir:  json文件路径
    :param lmdb_path:      lmdb存储地址
    :return:
    '''
    img_files = os.listdir(img_root_dir)
    omega = 3.
    db = lmdb.open(lmdb_path, map_size=int(1e12))

    with db.begin(write=True) as db_txn:
        for index, img_name in enumerate(img_files):
            image = cv2.imread(os.path.join(img_root_dir, img_name))
            json_path = os.path.join(json_root_dir, img_name[:-4] + '.json')
            if image is None or not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                js_data = json.load(f)
                for key in js_data.keys():
                    if key == u'shapes':
                        all_maps = []
                        for item_dict in js_data[key]:
                            tmp_feat_map_list = []
                            pts = item_dict[u'points']
                            if len(pts) != 4: break
                            x1, y1, x2, y2, x3, y3, x4, y4 = int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(
                                pts[1][1]), \
                                                             int(pts[2][0]), int(pts[2][1]), int(pts[3][0]), int(
                                pts[3][1])

                            for pt in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                                tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                                for r in range(tmp_feat_map.shape[0]):
                                    for c in range(tmp_feat_map.shape[1]):
                                        tmp_feat_map[r, c] = math.exp(
                                            -((c - pt[0]) * (c - pt[0]) + (r - pt[1]) * (r - pt[1])) / omega)
                                tmp_feat_map_list.append(tmp_feat_map)
                            all_maps.append(tmp_feat_map_list)

                        # 遍历获取最大值的feature map.
                        kpts_maps = []
                        arranged_maps = np.array(all_maps)  # [n, 4, h, w]
                        arranged_maps = arranged_maps.transpose([1, 0, 2, 3])

                        for maps in arranged_maps:
                            tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                            for r in range(tmp_feat_map.shape[0]):
                                for c in range(tmp_feat_map.shape[1]):
                                    tmp_feat_map[r, c] = max([xx[r, c] for xx in maps])
                            kpts_maps.append(tmp_feat_map)

                        if DEBUG_MODE:
                            for map in kpts_maps:
                                img_map = np.array(map * 255, dtype=np.uint8)
                                cv2.imshow("img", img_map)
                                cv2.waitKey()

                        kpts_maps = np.array(kpts_maps)
                        kpts_maps = kpts_maps.transpose([1, 2, 0])
                        # print(kpts_maps.shape)
                        input_array = np.zeros((image.shape[0], image.shape[1], 7), dtype=np.float32)
                        input_array[:, :, :3] = image
                        if PROCESS_IMG:
                            input_array[:, :, 0] -= 104.0
                            input_array[:, :, 0] *= 0.00390625
                            input_array[:, :, 1] -= 117.0
                            input_array[:, :, 1] *= 0.00390625
                            input_array[:, :, 2] -= 123.0
                            input_array[:, :, 2] *= 0.00390625
                        input_array[:, :, 3:] = kpts_maps
                        im_dat = caffe.io.array_to_datum(kpts_maps)
                        im_dat.label = 1
                        db_txn.put('{:0>10d}'.format(index).encode(), im_dat.SerializeToString())
    db.close()

def parse_from_xml(img_root_dir, xml_root_dir, lmdb_path, omega):
    '''
    从textbox格式的xml中解析并生成关键点热力图存入lmdb
    :param img_root_dir:
    :param xml_root_dir:
    :param lmdb_path:
    :return:
    '''
    img_files = os.listdir(img_root_dir)
    db = lmdb.open(lmdb_path, map_size=int(1e12))
    db_txn = db.begin(write=True)

    for index, img_name in enumerate(img_files):
        image = cv2.imread(os.path.join(img_root_dir, img_name))
        xml_path = os.path.join(xml_root_dir, img_name[:-4] + '.xml')
        if image is None or not os.path.exists(xml_path):
            continue

        reader = TextboxReader(xml_path)
        all_maps = []
        for poly in reader.get_objects():
            tmp_feat_map_list = []
            x1, y1, x2, y2, x3, y3, x4, y4 = int(poly[1][0]), int(poly[1][1]), int(poly[2][0]), int(
                poly[2][1]), \
                                             int(poly[3][0]), int(poly[3][1]), int(poly[4][0]), int(
                poly[4][1])
            for pt in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
                tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
                for r in range(tmp_feat_map.shape[0]):
                    for c in range(tmp_feat_map.shape[1]):
                        tmp_feat_map[r, c] = math.exp(
                            -((c - pt[0]) * (c - pt[0]) + (r - pt[1]) * (r - pt[1])) / omega)
                tmp_feat_map_list.append(tmp_feat_map)
            all_maps.append(tmp_feat_map_list)

        # 遍历获取最大值的feature map.
        kpts_maps = []
        arranged_maps = np.array(all_maps)  # [n, 4, h, w]
        arranged_maps = arranged_maps.transpose([1, 0, 2, 3])

        for maps in arranged_maps:
            tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for r in range(tmp_feat_map.shape[0]):
                for c in range(tmp_feat_map.shape[1]):
                    tmp_feat_map[r, c] = max([xx[r, c] for xx in maps])
            kpts_maps.append(tmp_feat_map)

        if DEBUG_MODE:
            for id, map in enumerate(kpts_maps):
                img_map = np.array(map*255, dtype=np.uint8)
                cv2.imwrite(os.path.join(img_root_dir, img_name[:-4]+'-'+str(id)+'.jpg'), img_map)
                # cv2.imshow("img", img_map)
                # cv2.imshow("src", image)
                # cv2.waitKey()

        kpts_maps = np.array(kpts_maps)
        kpts_maps = kpts_maps.transpose([1, 2, 0])
        input_array = np.zeros((image.shape[0], image.shape[1], 7), dtype=np.float32)
        input_array[:, :, :3] = image
        if PROCESS_IMG:
            input_array[:, :, 0] -= 104.0
            input_array[:, :, 0] *= 0.00390625
            input_array[:, :, 1] -= 117.0
            input_array[:, :, 1] *= 0.00390625
            input_array[:, :, 2] -= 123.0
            input_array[:, :, 2] *= 0.00390625
        input_array[:, :, 3:] = kpts_maps
        im_dat = caffe.io.array_to_datum(kpts_maps)
        im_dat.label = 1
        db_txn.put('{:0>10d}'.format(index).encode(), im_dat.SerializeToString())

        if index % 10 == 0:
            db_txn.commit()
            db_txn = db.begin(write=True)
        print('Processed image %s.' % img_name)

    db.close()

def sinle_processs_xml(img_name, img_root_dir, xml_root_dir, q, normlize = True):
    global COUNTER
    image = cv2.imread(os.path.join(img_root_dir, img_name))
    xml_path = os.path.join(xml_root_dir, img_name[:-4] + '.xml')
    if image is None or not os.path.exists(xml_path):
        q.put('INVALID')

    reader = TextboxReader(xml_path)
    all_maps = []
    for poly in reader.get_objects():
        tmp_feat_map_list = []
        x1, y1, x2, y2, x3, y3, x4, y4 = int(poly[1][0]), int(poly[1][1]), int(poly[2][0]), int(
            poly[2][1]), \
                                         int(poly[3][0]), int(poly[3][1]), int(poly[4][0]), int(
            poly[4][1])
        for pt in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]:
            tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for r in range(tmp_feat_map.shape[0]):
                for c in range(tmp_feat_map.shape[1]):
                    tmp_feat_map[r, c] = math.exp(
                        -((c - pt[0]) * (c - pt[0]) + (r - pt[1]) * (r - pt[1])) / omega)
            tmp_feat_map_list.append(tmp_feat_map)
        all_maps.append(tmp_feat_map_list)

    # 遍历获取最大值的feature map.
    kpts_maps = []
    arranged_maps = np.array(all_maps)  # [n, 4, h, w]
    arranged_maps = arranged_maps.transpose([1, 0, 2, 3])

    for maps in arranged_maps:
        tmp_feat_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        for r in range(tmp_feat_map.shape[0]):
            for c in range(tmp_feat_map.shape[1]):
                tmp_feat_map[r, c] = max([xx[r, c] for xx in maps])
        kpts_maps.append(tmp_feat_map)

    kpts_maps = np.array(kpts_maps)
    kpts_maps = kpts_maps.transpose([1, 2, 0])
    input_array = np.zeros((image.shape[0], image.shape[1], 7), dtype=np.float32)
    input_array[:, :, :3] = image
    if normlize:
        input_array[:, :, 0] -= 104.0
        input_array[:, :, 0] *= 0.00390625
        input_array[:, :, 1] -= 117.0
        input_array[:, :, 1] *= 0.00390625
        input_array[:, :, 2] -= 123.0
        input_array[:, :, 2] *= 0.00390625
    input_array[:, :, 3:] = kpts_maps
    im_dat = caffe.io.array_to_datum(kpts_maps)
    im_dat.label = 1

    # with LOCK:
    #     COUNTER.value += 1
    #     db_txn.put('{:0>10d}'.format(COUNTER).encode(), im_dat.SerializeToString())
    #     print('Processed image %s.' % '{:0>10d}'.format(COUNTER))
    q.put(im_dat.SerializeToString())

def start_listen(q, dbpath):
    global COUNTER
    db = lmdb.open(dbpath, map_size=int(1e12))
    db_txn = db.begin(write=True)
    while 1:
        # if q.empty():continue
        imgstr = q.get()
        if imgstr == STOP_TOKEN:
            break
        elif imgstr == 'INVALID':
            continue
        try:
            db_txn.put('{:0>10d}'.format(COUNTER).encode(), imgstr.get())
            COUNTER.value += 1
            print('Processed image %s.' % '{:0>10d}'.format(COUNTER))
        except:
            traceback.print_exc()
        if COUNTER.value % 5 == 0:
            db_txn.commit()
            db_txn = db.begin(write=True)
    db.close()

if __name__ == '__main__':
    img_root_dir = '/home/wz/DataSets/id_card/camera/color/aug/img_aug'
    xml_root_dir = '/home/wz/DataSets/id_card/camera/color/aug/xml_aug'
    dbpath = '/home/wz/DataSets/id_card/camera/color/aug/lmdb'
    omega = 3.
    '''
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(processes=None)
    pool.apply_async(start_listen(q, dbpath))
    res = [pool.apply_async(sinle_processs_xml, args=(imgname, img_root_dir, xml_root_dir, q, True))
           for imgname in os.listdir(img_root_dir)]
    pool.join()
    pool.close()
    '''
    parse_from_xml(img_root_dir, xml_root_dir, dbpath, omega)