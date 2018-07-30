#-- coding:utf-8 --

from pascal_voc_io import  *
import os
import cv2
import numpy as np
import random

###################################################################
# 用于将前后制作的行驶证训练图片做融合
###################################################################

if __name__ == '__main__':
    ori_xml_dir = '/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped/xml_split_0'
    ori_img_dir = '/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped/img_split_0'

    res_xml_dir = '/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped/VOC2007/Annotations'
    res_img_dir = '/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped/VOC2007/JPEGImages'

    xml_names = os.listdir(ori_xml_dir)
    img_names = os.listdir(ori_img_dir)

    # 1st step, filter OTHER label
    '''
    for xml in xml_names:
        img_name = (xml[:-4]+'.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        xml_path = os.path.join(ori_xml_dir, xml)
        xml_reader = PascalVocReader(xml_path)
        xml_writer = PascalVocWriter(foldername=ori_img_dir, filename=xml[:-4],imgSize = (xml_reader.height,xml_reader.width))
        OTHER_NUM = 0
        right_most_pt = [0,0]
        VIN_NUM = 0
        ENG_NUM = 0
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            if label == 'OTHER':
                OTHER_NUM += 1
                right_most_pt = right_top
            elif label == 'VIN':
                right_most_pt = right_top
                VIN_NUM += 1
            else:
                ENG_NUM += 1

        # 存右边子图，并修改xml坐标
        if ENG_NUM > 15:
            cv2.imwrite(os.path.join(res_img_dir, img_name[:-4] + '_0.jpg'), ori_img[:, right_most_pt[0]:, :])
        else:
            cv2.imwrite(os.path.join(res_img_dir, img_name), ori_img[:, right_most_pt[0]:, :])
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]
            label = shape[0]
            if label == 'OTHER' or label == "VIN": continue
            xml_writer.addBndBox(int(left_top[0]) - right_most_pt[0],
                                 int(left_top[1]),
                                 int(right_bottom[0] - right_most_pt[0]),
                                 int(right_bottom[1]),
                                 label)
        if ENG_NUM > 15:
            xml_writer.save(os.path.join(res_xml_dir, img_name[:-4] + '_0.xml'))
        else:
            xml_writer.save(os.path.join(res_xml_dir, img_name[:-4] + '.xml'))
    # 2nd step, filter split long img into two.
    '''

    # 2nd step, split lone images into two.
    '''
    for xml in xml_names:
        img_name = (xml[:-4] + '.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        xml_path = os.path.join(ori_xml_dir, xml)
        xml_reader = PascalVocReader(xml_path)

        if len(xml_reader.getShapes()) < 11: continue
        rects = []
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            rects.append(left_top)

        shapes = np.array(xml_reader.getShapes())
        rects = np.array(rects)
        sort_index = rects[:, 0].argsort()
        rects = rects[sort_index]
        shapes = shapes[sort_index]

        # print(rects)
        left_sub_img = ori_img[:, 0: int(rects[10, 0]), ]
        right_sub_img = ori_img[:, int(rects[-10, 0]):, ]

        cv2.imwrite(os.path.join(res_img_dir, img_name[:-4]) + '_left.jpg', left_sub_img)
        xml_writer_left = PascalVocWriter(foldername=ori_img_dir, filename=xml[:-4]+'_left',
                                     imgSize = left_sub_img.shape)
        for shape in shapes[0:10]:
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            xml_writer_left.addBndBox(left_top[0],left_top[1],right_bottom[0],right_bottom[1],label)
        xml_writer_left.save(os.path.join(res_xml_dir, img_name[:-4]) + '_left.xml')

        cv2.imwrite(os.path.join(res_img_dir, img_name[:-4]) + '_right.jpg', right_sub_img)
        xml_writer_right = PascalVocWriter(foldername = ori_img_dir, filename = xml[:-4]+'_right',
                                          imgSize=right_sub_img.shape)
        print(shapes)
        start_x = rects[-10, 0]
        for shape in shapes[-10:]:
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            xml_writer_right.addBndBox(left_top[0] - start_x, left_top[1], right_bottom[0] - start_x, right_bottom[1], label)
        xml_writer_right.save(os.path.join(res_xml_dir, img_name[:-4]) + '_right.xml')
        '''

    # 3rd step, split lone images of textbox into two.
    '''
    for xml in xml_names:
        img_name = (xml[:-4] + '.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        xml_path = os.path.join(ori_xml_dir, xml)
        xml_reader = PascalVocReader(xml_path)

        rects = []
        rt_bts = []
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            if label == 'OTHER': continue
            rects.append(left_top)
            rt_bts.append(right_bottom)

        if len(rects) < 15: continue
        print(len(rects))
        shapes = np.array(xml_reader.getShapes())
        rects = np.array(rects)
        rt_bts = np.array(rt_bts)
        sort_index = rects[:, 0].argsort()
        rects = rects[sort_index]
        shapes = shapes[sort_index]
        rt_bts = rt_bts[sort_index]

        left_top = rects[0] - [20, 20]
        right_bottom = rt_bts[-1] + [20, 20]
        left_top[0] = max(left_top[0], 0)
        left_top[1] = max(left_top[1], 0)
        right_bottom[0] = min(right_bottom[0], ori_img.shape[1] - 1)
        right_bottom[1] = min(right_bottom[1], ori_img.shape[0] - 1)

        # sub_img = ori_img[left_top[1] : right_bottom[1], left_top[0] : right_bottom[0]]
        # cv2.imshow('subimg', sub_img)
        # cv2.waitKey(0)

        left_sub_img = ori_img[left_top[1]: right_bottom[1], left_top[0]: rects[10][0], :]
        right_sub_img = ori_img[left_top[1]: right_bottom[1], rects[-10, 0] : right_bottom[0], :]

        cv2.imwrite(os.path.join(res_img_dir, img_name[:-4]) + '_left.jpg', left_sub_img)
        xml_writer_left = PascalVocWriter(foldername=ori_img_dir, filename=xml[:-4] + '_left',
                                          imgSize=left_sub_img.shape)
        for shape in shapes[0:10]:
            points = shape[1]
            lefttop = points[0]
            right_top = points[1]
            rightbottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            xml_writer_left.addBndBox(lefttop[0] - left_top[0], lefttop[1] - left_top[1], rightbottom[0] - left_top[0],
                                      rightbottom[1] - left_top[1], label)
        xml_writer_left.save(os.path.join(res_xml_dir, img_name[:-4]) + '_left.xml')

        cv2.imwrite(os.path.join(res_img_dir, img_name[:-4]) + '_right.jpg', right_sub_img)
        xml_writer_right = PascalVocWriter(foldername=ori_img_dir, filename=xml[:-4] + '_right',
                                           imgSize=right_sub_img.shape)
        start_x = rects[-10, 0]
        for shape in shapes[-10:]:
            points = shape[1]
            lefttop = points[0]
            right_top = points[1]
            rightbottom = points[2]
            left_bottom = points[3]

            label = shape[0]
            xml_writer_right.addBndBox(lefttop[0] - rects[-10][0], lefttop[1] - left_top[1],
                                       rightbottom[0] - rects[-10][0], rightbottom[1] - left_top[1],
                                       label)
        xml_writer_right.save(os.path.join(res_xml_dir, img_name[:-4]) + '_right.xml')
        '''

    # 3.5 step, transform into 320*1056
    '''
    for xml in xml_names:
        xml_path = os.path.join(ori_xml_dir, xml)
        img_name = (xml[:-4]+'.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        height = ori_img.shape[0]
        width = ori_img.shape[1]
        if width > 1056:
            continue

        scale = width * 1.0 / height

        standard_width = 1056
        standard_height =int(standard_width / scale)

        hori_border = True
        if standard_height > 320:
            scale = height * 1.0 / width
            standard_height = 320
            standard_width = int(standard_height / scale)
            hori_border = False

        res_img = np.zeros((320, 1056, 3), dtype=np.uint8) + 255
        standard_img = cv2.resize(ori_img, dsize=(standard_width, standard_height),interpolation=cv2.INTER_CUBIC)
        margin = 0
        if hori_border:
            margin = (320 - standard_height) / 2
            res_img[margin:margin + standard_height, :, :] = standard_img
        else:
            margin = (1056 - standard_width) / 2
            res_img[:, margin:margin + standard_width, :] = standard_img
        cv2.imwrite(os.path.join(res_img_dir, xml[:-4] + '.jpg'), res_img)

        xml_reader = PascalVocReader(xml_path)
        xml_writer = PascalVocWriter(foldername=ori_img_dir, filename=xml[:-4], imgSize=res_img.shape)

        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]

            label = shape[0]

            zoomout_scale_w = standard_width * 1.0 / width
            zoomout_scale_h = standard_height * 1.0 / height

            if hori_border:
                xml_writer.addBndBox(int(left_top[0] * zoomout_scale_w),
                                     int(left_top[1] * zoomout_scale_h + margin),
                                     int(right_bottom[0] * zoomout_scale_w),
                                     int(right_bottom[1] * zoomout_scale_h + margin),
                                     label)
            else:
                xml_writer.addBndBox(int(left_top[0] * zoomout_scale_w + margin),
                                     int(left_top[1] * zoomout_scale_h),
                                     int(right_bottom[0] * zoomout_scale_w + margin),
                                     int(right_bottom[1] * zoomout_scale_h),
                                     label)

        xml_writer.save(os.path.join(res_xml_dir, xml[:-4] + XML_EXT))
        print('Processed img %s.' % xml)
        '''

    # 4th step, randomly group images into pvasahpe.
    '''
    id = 0
    for i in range(5):
        for xml in xml_names:
            pair_index = random.randint(0, len(xml_names) - 1)
            xml_path = os.path.join(ori_xml_dir, xml)
            pair_xml_path = os.path.join(ori_xml_dir, xml_names[pair_index])
            xml_reader = PascalVocReader(xml_path)
            pair_xml_reader = PascalVocReader(pair_xml_path)

            img_name = (xml[:-4]+'.jpg')
            pair_img_name = (xml_names[pair_index][:-4] + '.jpg')
            if not img_name in img_names or not pair_img_name in img_names:
                continue
            img_path = os.path.join(ori_img_dir, img_name)
            ori_img = cv2.imread(img_path)
            pair_img_path = os.path.join(ori_img_dir, pair_img_name)
            pair_ori_img = cv2.imread(pair_img_path)

            stitch_img = np.zeros(shape=(640, 1056, 3),dtype = np.uint8) + 255
            stitch_img[0:320, :, :] = ori_img
            stitch_img[320:, : , :] = pair_ori_img
            cv2.imwrite(os.path.join(res_img_dir, str(id)+'.jpg'), stitch_img)


            xml_writer = PascalVocWriter(foldername = ori_img_dir, filename = str(id),
                                         imgSize = stitch_img.shape)

            for shape in xml_reader.getShapes():
                points = shape[1]
                left_top = points[0]
                right_top = points[1]
                right_bottom = points[2]
                left_bottom = points[3]

                label = shape[0]
                xml_writer.addBndBox(int(left_top[0]),
                                     int(left_top[1]),
                                     int(right_bottom[0]),
                                     int(right_bottom[1]),
                                     label)
            for shape in pair_xml_reader.getShapes():
                points = shape[1]
                left_top = points[0]
                right_top = points[1]
                right_bottom = points[2]
                left_bottom = points[3]

                label = shape[0]
                xml_writer.addBndBox(int(left_top[0]),
                                     int(left_top[1]) + 320,
                                     int(right_bottom[0]),
                                     int(right_bottom[1]  + 320),
                                     label)

            xml_writer.save(os.path.join(res_xml_dir, str(id) + XML_EXT))
            id += 1
        '''

    # 5th step, remove boxes which out of the image, use 'filterEmptyFile' script.