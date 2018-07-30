#-- coding:utf-8 --

from pascal_voc_io import  *
import os
import cv2
import numpy as np


###################################################################
#
# 将按照单个字符标注的VOC数据集按照单行字符串裁剪并写入VOC格式。
#
###################################################################

if __name__ == '__main__':
    ori_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/chars_on_ori_img/xml'
    ori_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/chars_on_ori_img/img'

    res_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/chars_on_ori_img/single_char/'
    res_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/chars_on_ori_img/single_char/'

    xml_names = os.listdir(ori_xml_dir)
    img_names = os.listdir(ori_img_dir)

    for xml in xml_names:
        xml_path = os.path.join(ori_xml_dir, xml)
        img_name = (xml[:-4]+'.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)
        # cv2.imshow('imgpath', ori_img)
        # cv2.waitKey(0)

        xml_reader = PascalVocReader(xml_path)
        shapes = xml_reader.getShapes()
        # print(shapes)
        visited_flag = np.zeros(len(shapes), dtype=np.uint8)
        # print type(shapes)

        #先按照x排序，再按照y排序，然后遍历切割即可
        rect_coors = [[x[1][0][0],x[1][0][1],x[1][2][0], x[1][2][1]] for x in shapes]#左上角和右下角点
        labels = [x[0] for x in shapes]

        rect_coors = np.array(rect_coors)
        labels = np.array(labels, dtype=np.str)
        x_sort_index = rect_coors[:,0].argsort() #记录下索引方便对应取label
        rect_coors = rect_coors[x_sort_index]    #按x排序，保持y与x对应关系不变，注意argsort()返回的只是索引
        labels = labels[x_sort_index]
        # y_sort_index = rect_coors[:,1].argsort()
        # rect_coors = rect_coors[y_sort_index]    #按y排序，保持y与x对应关系不变，注意argsort()返回的只是索引
        # labels = labels[y_sort_index]            #确保所有都遍历还是加一个flag

        #遍历写出到xml文件
        lineid = 0
        padding  = 12
        for id in range(rect_coors.shape[0]):
            if visited_flag[id] == 1: continue

            cur_lefttop_pt = [rect_coors[id][0], rect_coors[id][1]]
            cur_rtbt_pt = [rect_coors[id][2], rect_coors[id][3]]

            line_candi_ids = []
            line_candi_ids.append(id)
            last_rtbt_pt = cur_rtbt_pt

            # 遍历其它rect，如果x值和y值接近则加入候选框中
            for id_inner in range(rect_coors.shape[0]):
                if id_inner == id: continue
                if visited_flag[id_inner] == 1: continue
                if abs(rect_coors[id_inner][0] - last_rtbt_pt[0]) < 10 and \
                        abs(rect_coors[id_inner][3] - last_rtbt_pt[1]) < 10:
                    last_rtbt_pt = [rect_coors[id_inner][2], rect_coors[id_inner][3]]
                    line_candi_ids.append(id_inner)

            lefttop = np.array(cur_lefttop_pt) - padding
            rtbt = np.array([rect_coors[line_candi_ids[-1]][2], rect_coors[line_candi_ids[-1]][3]]) + padding

            lefttop[0] = max(lefttop[0], 0)
            lefttop[1] = max(lefttop[1], 0)
            rtbt[0] = min(rtbt[0], ori_img.shape[1] - 1)
            rtbt[1] = min(rtbt[1], ori_img.shape[0] - 1)

            subline = ori_img[lefttop[1]:rtbt[1], lefttop[0]:rtbt[0]]
            if subline.shape[1] * 1.0 / subline.shape[0] <= 4: #filter long lines.
                continue
            cv2.imwrite(os.path.join(res_img_dir, xml[:-4] + '_' + str(lineid) + '.jpg'), subline)
            xml_writer = PascalVocWriter(foldername = ori_img_dir, filename = xml[:-4] + '_' + str(lineid),
                                         imgSize = subline.shape)
            for i in line_candi_ids:
                xml_writer.addBndBox(rect_coors[i][0] - lefttop[0], rect_coors[i][1] - lefttop[1], rect_coors[i][2] - lefttop[0],
                                     rect_coors[i][3] - lefttop[1], labels[i])
                visited_flag[i] = 1
            xml_writer.save(os.path.join(res_xml_dir, xml[:-4] + '_' + str(lineid) + XML_EXT))

            lineid += 1
            print('Processed %d lines.' % lineid)