import json
import codecs
import os
import cv2
import random

from textbox_io import *


def transform_json_to_xml():
    # data root dir includes json file and images.
    data_root = '/home/wz/DataSets/LICENCES/invoice_FaPiao/pix2pix/merge'
    save_dir = '/home/wz/DataSets/LICENCES/invoice_FaPiao/pix2pix/xml'

    file_names = os.listdir(data_root)
    for file_name in file_names:
        if file_name.endswith('.json'):
            image = cv2.imread(os.path.join(data_root, file_name[:-5] + '.jpg'))
            if image is None:
                continue
            with open(os.path.join(data_root, file_name), 'r') as f:
                js_data = json.load(f)

                xml_writer = TextboxWriter(None, None, None)
                xml_writer.set_foldername('Carbrand')
                xml_writer.set_size(image.shape)

                for key in js_data.keys():
                    if key == u'shapes':
                        for item_dict in js_data[key]:
                            pts = item_dict[u'points']
                            if len(pts) != 4: break
                            x1, y1, x2, y2, x3, y3, x4, y4 = int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(
                                pts[1][1]), \
                                                             int(pts[2][0]), int(pts[2][1]), int(pts[3][0]), int(
                                pts[3][1])

                            xmin = min(x1, x2, x3, x4)
                            xmax = max(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            ymax = max(y1, y2, y3, y4)
                            xml_writer.add_bndbox(x1, y1, x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax, 'carbrand')

                    if key == 'imagePath':
                        xml_writer.set_filename(js_data[key])
                xml_writer.save(os.path.join(save_dir, file_name[:-5] + '.xml'))


def gen_train_testset():
    xml_root_path = '/home/wz/Data/VIN/textbox_train_data/aug/xml_aug'
    img_root_path = '/home/wz/Data/VIN/textbox_train_data/aug/img_aug'
    xmls = os.listdir(xml_root_path)
    random.shuffle(xmls)
    test_ratio = 0.1
    train_xmls = xmls[:int(test_ratio*len(xmls))]
    test_xmls = xmls[int(test_ratio*len(xmls)):]
    with open('train.txt', 'w') as fout:
        for xml in train_xmls:
            fout.writelines(os.path.join(img_root_path, xml[:-4]+'.jpg') + ' '+
                            os.path.join(xml_root_path, xml) + '\n')
    with open('test.txt', 'w') as fout:
        for xml in test_xmls:
            fout.writelines(os.path.join(img_root_path, xml[:-4]+'.jpg') + ' '+
                            os.path.join(xml_root_path, xml) + '\n')


if __name__ == '__main__':
    transform_json_to_xml()
    # gen_train_testset()