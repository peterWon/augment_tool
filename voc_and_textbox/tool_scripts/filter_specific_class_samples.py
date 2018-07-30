import sys
sys.path.append("..")
from pascal_voc_io import *
import os
import shutil

xml_folder = '/home/wz/Desktop/VOC2007/Annotations'
image_folder = '/home/wz/Desktop/VOC2007/JPEGImages'
filtered_xml_folder = '/home/wz/Desktop/VOC2007/ForegroundBackGround/Annotations'
filtered_image_folder = '/home/wz/Desktop/VOC2007/ForegroundBackGround/JPEGImages'


for xml_name in os.listdir(xml_folder):
    xml_reader = PascalVocReader(os.path.join(xml_folder, xml_name))
    to_filter = False
    for shape in xml_reader.getShapes():
        label = shape[0]
        if label == "QueChaoKaFei" or label == "LeShiShuPian" or label == "HuangGuanQuQi":
            to_filter = True

    if not to_filter:
        tmp = PascalVocWriter(filtered_xml_folder, xml_name[:-4], [xml_reader.width, xml_reader.height])
        for shape in xml_reader.getShapes():
            points = shape[1]
            left_top = points[0]
            right_top = points[1]
            right_bottom = points[2]
            left_bottom = points[3]
            tmp.addBndBox(int(left_top[0]), int(left_top[1]), int(right_bottom[0]), int(right_bottom[1]),
                          str('foreground'))
        tmp.save(filtered_xml_folder + "/" + xml_name[:-4] + XML_EXT)
        shutil.copy(os.path.join(image_folder, xml_name[:-4] + '.jpg'),
                    os.path.join(filtered_image_folder, xml_name[:-4] + '.jpg'))

print('Done!')