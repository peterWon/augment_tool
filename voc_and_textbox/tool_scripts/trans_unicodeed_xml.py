# -*- coding:utf-8 -*-
'''
把unicode标签的xml转换到int标签
'''
import os
import multiprocessing as mp
from pascal_voc_io import *

def single_process(xml):
    if not os.path.exists(os.path.join("/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages", xml[:-4]+'.jpg')):return
    reader = PascalVocReader(os.path.join(src_xml_dir, xml))
    size = (reader.height, reader.width, 3)
    writer = PascalVocWriter(dst_xml_dir, xml[:-4], size)
    for shape in reader.getShapes():
        points = shape[1]
        left_top = points[0]
        right_top = points[1]
        right_bottom = points[2]
        left_bottom = points[3]
        label = labelmap[shape[0]]
        writer.addBndBox(int(left_top[0]),
                             int(left_top[1]),
                             int(right_bottom[0]),
                             int(right_bottom[1]),
                             str(label))
    writer.save(os.path.join(dst_xml_dir, xml))

if __name__ == '__main__':
    # src_xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations_600000_unicode'
    # dst_xml_dir = '/home/wz/DataSets/SYNTH_LINE/VOC2007/Annotations'
    # labeltxt = '/home/wz/DataSets/SYNTH_LINE/classes_name_unicode.txt'
    # labelmap = {}
    # with codecs.open(labeltxt, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i, l in enumerate(lines):
    #         labelmap.update({l.strip(): i})
    #
    # print(labelmap)
    #
    # xmls = os.listdir(src_xml_dir)
    # pool = mp.Pool(None)
    # pool.map(single_process, xmls)



#生成int labelindex
# with open('/home/wz/DataSets/SYNTH_LINE/classes_name.txt', 'w') as f:
#     for i in range(5071):
#         f.writelines(str(i)+'\n')


#append label to caffeocr.
# with open("/home/wz/DataSets/SYNTH_LINE/VOC2007/tmp_labels.txt", 'r') as f:
#     lines = f.readlines()
#     with open("/home/wz/DataSets/SYNTH_LINE/VOC2007/all_labels.txt", 'w') as of:
#         for l in lines:
#             base_name, label = l.strip().split(" ")
#
#             binary_base_name = str(int(base_name) + 200000)
#             if os.path.exists(os.path.join("/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages", binary_base_name+'.jpg')):
#                 of.writelines(binary_base_name+" "+label+"\n")
#
#             fake_base_name = str(int(base_name) + 400000)
#             if os.path.exists(
#                     os.path.join("/home/wz/DataSets/SYNTH_LINE/VOC2007/JPEGImages", fake_base_name + '.jpg')):
#                 of.writelines(fake_base_name + " " + label + "\n")
#
#             print(base_name, binary_base_name, fake_base_name)
#             of.writelines(l)

    with open('/home/wz/DeepLearning/caffe_dir/easy-pvanet-unicode/data/VOCdevkit2007/classes_name.txt', 'w') as f:
        for i in range(5071):
            f.writelines(str(i)+'\n')