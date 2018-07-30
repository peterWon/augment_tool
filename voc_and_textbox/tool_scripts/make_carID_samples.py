#-- coding:utf-8 --
import sys
sys.path.append("..")

from pascal_voc_io import  *
import os
import cv2
import numpy as np
import random

###################################################################
#
# 制作行驶证车辆识别代号样本
# 策略：
# 1）进入的图片先按照固定长宽比缩放（640×1056）
# 2）尽量保持字符的形状比例，不足的地方补黑(或者补白)
###################################################################
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
        margin = (to_height - standard_height) / 2
        res_img[margin:margin + standard_height, :, :] = standard_img
    else:
        margin = (to_width - standard_width) / 2
        res_img[:, margin:margin + standard_width, :] = standard_img
    zoomout_scale_w = standard_width * 1.0 / width
    zoomout_scale_h = standard_height * 1.0 / height

    return res_img, zoomout_scale_h,zoomout_scale_w,margin,hori_border


def reshape_img_and_write_xml(to_height = 640, to_width = 1056):
    ori_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Annotations'
    ori_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/JPEGImages'

    res_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Reshaped_For_Mb_SSD/Annotations'
    res_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Reshaped_For_Mb_SSD/JPEGImages'

    xml_names = os.listdir(ori_xml_dir)
    img_names = os.listdir(ori_img_dir)

    for xml in xml_names:
        xml_path = os.path.join(ori_xml_dir, xml)
        img_name = (xml[:-4] + '.jpg')
        if not img_name in img_names:
            continue

        img_path = os.path.join(ori_img_dir, img_name)
        ori_img = cv2.imread(img_path)

        height = ori_img.shape[0]
        width = ori_img.shape[1]
        if width > to_width:
            continue

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
            margin = (to_height - standard_height) / 2
            res_img[margin:margin + standard_height, :, :] = standard_img
        else:
            margin = (to_width - standard_width) / 2
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

def stitch_imgs_to_pvashape():
    ori_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Annotations'
    ori_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/JPEGImages'

    res_xml_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Stitch_For_PVA/Annotations'
    res_img_dir = '/home/wz/Data/LabelingAlfa/number_and_char/split_shorter_datasets/Stitch_For_PVA/JPEGImages'

    xml_names = os.listdir(ori_xml_dir)
    img_names = os.listdir(ori_img_dir)

    IMAGE_EXT = '.jpg'

    # step 1, read xmls and get number of chars they have,then put them in different structure.
    xml_longer = []  # 整车型号，车模型
    xml_shorter = []
    for xml in xml_names:
        xml_path = os.path.join(ori_xml_dir, xml)
        img_name = (xml[:-4] + IMAGE_EXT)
        if not img_name in img_names:
            continue

        xml_reader = PascalVocReader(xml_path)
        if len(xml_reader.getShapes()) == 8:
            xml_longer.append(xml)
        elif len(xml_reader.getShapes()) < 8:
            xml_shorter.append(xml)

    # step 2, put them in pva canvas and write xmls out.
    random.shuffle(xml_shorter)
    random.shuffle(xml_longer)

    for i in range(2000):
        long_sample_1_xml = random.choice(xml_longer)
        long_sample_2_xml = random.choice(xml_longer)

        short_sample_1_xml = random.choice(xml_shorter)
        short_sample_2_xml = random.choice(xml_shorter)
        short_sample_3_xml = random.choice(xml_shorter)
        short_sample_4_xml = random.choice(xml_shorter)

        long_sample_1_img = cv2.imread(os.path.join(ori_img_dir, long_sample_1_xml[:-4] + IMAGE_EXT))
        long_sample_2_img = cv2.imread(os.path.join(ori_img_dir, long_sample_2_xml[:-4] + IMAGE_EXT))
        short_sample_1_img = cv2.imread(os.path.join(ori_img_dir, short_sample_1_xml[:-4] + IMAGE_EXT))
        short_sample_2_img = cv2.imread(os.path.join(ori_img_dir, short_sample_2_xml[:-4] + IMAGE_EXT))
        short_sample_3_img = cv2.imread(os.path.join(ori_img_dir, short_sample_3_xml[:-4] + IMAGE_EXT))
        short_sample_4_img = cv2.imread(os.path.join(ori_img_dir, short_sample_4_xml[:-4] + IMAGE_EXT))

        pvaimg = np.zeros(shape=(640, 1056, 3), dtype=np.uint8)
        long_sample_1_img, sc_h_0, sc_w_0, margin_0, hori_0 = reshape_pva(long_sample_1_img, 320, 528)
        long_sample_2_img, sc_h_1, sc_w_1, margin_1, hori_1 = reshape_pva(long_sample_2_img, 320, 528)
        short_sample_1_img, sc_h_2, sc_w_2, margin_2, hori_2 = reshape_pva(short_sample_1_img, 320, 264)
        short_sample_2_img, sc_h_3, sc_w_3, margin_3, hori_3 = reshape_pva(short_sample_2_img, 320, 264)
        short_sample_3_img, sc_h_4, sc_w_4, margin_4, hori_4 = reshape_pva(short_sample_3_img, 320, 264)
        short_sample_4_img, sc_h_5, sc_w_5, margin_5, hori_5 = reshape_pva(short_sample_4_img, 320, 264)

        #长条放在左、中、右
        mode = random.choice([0, 1, 2])
        if mode == 0:
            pvaimg[:320, :528] = long_sample_1_img
            pvaimg[320:, :528] = long_sample_2_img
            pvaimg[:320, 528:792] = short_sample_1_img
            pvaimg[320:, 528:792] = short_sample_2_img
            pvaimg[:320, 792:] = short_sample_3_img
            pvaimg[320:, 792:] = short_sample_4_img
        elif mode == 1:
            pvaimg[:320, 264:792] = long_sample_1_img
            pvaimg[320:, 264:792] = long_sample_2_img
            pvaimg[:320, :264] = short_sample_1_img
            pvaimg[320:, :264] = short_sample_2_img
            pvaimg[:320, 792:] = short_sample_3_img
            pvaimg[320:, 792:] = short_sample_4_img
        else:
            pvaimg[:320, 528:] = long_sample_1_img
            pvaimg[320:, 528:] = long_sample_2_img
            pvaimg[:320, :264] = short_sample_1_img
            pvaimg[320:, :264] = short_sample_2_img
            pvaimg[:320, 264:528] = short_sample_3_img
            pvaimg[320:, 264:528] = short_sample_4_img

        cv2.imwrite(os.path.join(res_img_dir, str(i) + '.jpg'), pvaimg)
        xml_writer = PascalVocWriter(foldername=ori_img_dir, filename=str(i),
                                     imgSize=pvaimg.shape)

        sc_w = [sc_w_0, sc_w_1, sc_w_2, sc_w_3, sc_w_4, sc_w_5]
        sc_h = [sc_h_0, sc_h_1, sc_h_2, sc_h_3, sc_h_4, sc_h_5]
        margin = [margin_0, margin_1, margin_2, margin_3, margin_4, margin_5]
        hori = [hori_0, hori_1, hori_2, hori_3, hori_4, hori_5]

        for index, xml in enumerate([long_sample_1_xml, long_sample_2_xml,
                                     short_sample_1_xml, short_sample_2_xml,
                                     short_sample_3_xml, short_sample_4_xml]):
            xml_reader = PascalVocReader(os.path.join(ori_xml_dir, xml))
            for shape in xml_reader.getShapes():
                points = shape[1]
                left_top = points[0]
                right_top = points[1]
                right_bottom = points[2]
                left_bottom = points[3]
                label = shape[0]
                offset = [0, 0]  # (x,y)
                if mode == 0:
                    if index == 0:
                        offset = [0, 0]
                    elif index == 1:
                        offset = [0, 320]
                    elif index == 2:
                        offset = [528, 0]
                    elif index == 3:
                        offset = [528, 320]
                    elif index == 4:
                        offset = [792, 0]
                    elif index == 5:
                        offset = [792, 320]
                elif mode == 1:
                    if index == 0:
                        offset = [264, 0]
                    elif index == 1:
                        offset = [264, 320]
                    elif index == 2:
                        offset = [0, 0]
                    elif index == 3:
                        offset = [0, 320]
                    elif index == 4:
                        offset = [792, 0]
                    elif index == 5:
                        offset = [792, 320]
                else:
                    if index == 0:
                        offset = [528, 0]
                    elif index == 1:
                        offset = [528, 320]
                    elif index == 2:
                        offset = [0, 0]
                    elif index == 3:
                        offset = [0, 320]
                    elif index == 4:
                        offset = [264, 0]
                    elif index == 5:
                        offset = [264, 320]

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
        xml_writer.save(os.path.join(res_xml_dir, str(i) + XML_EXT))

        print('Saved %d images successfully!' % i)


if __name__ == '__main__':
    # stitch_imgs_to_pvashape()
    reshape_img_and_write_xml(to_width=300,to_height=300)