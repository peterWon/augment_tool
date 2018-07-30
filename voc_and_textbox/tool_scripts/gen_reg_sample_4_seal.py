#-- coding:utf-8 --
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np
import copy
import os

XML_EXT = '.xml'

class TextboxReader:
    def __init__(self, filepath):
        self.polygons = []
        self.filepath = filepath
        self.verified = False
        self.image_width = 0
        self.image_height = 0
        self.parseXML()

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    # parse XML to get polygons
    def parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.root = xmltree
        filename = xmltree.find('filename').text

        size = xmltree.find('size')
        self.image_width = int(size.find('width').text)
        self.image_height = int(size.find('height').text)
        print('image height, image width:', self.image_height, self.image_width)

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find('bndbox')
            self.addPolygon(bndbox)
        return True

    # get polygons
    def getPolygons(self):
        return self.polygons

    # form xml get polygon and append it in polygons
    def addPolygon(self, polygon):
        x1 = eval(polygon.find('x1').text)
        y1 = eval(polygon.find('y1').text)
        x2 = eval(polygon.find('x2').text)
        y2 = eval(polygon.find('y2').text)
        x3 = eval(polygon.find('x3').text)
        y3 = eval(polygon.find('y3').text)
        x4 = eval(polygon.find('x4').text)
        y4 = eval(polygon.find('y4').text)
        xmin = eval(polygon.find('xmin').text)
        ymin = eval(polygon.find('ymin').text)
        xmax = eval(polygon.find('xmax').text)
        ymax = eval(polygon.find('ymax').text)
        points = [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(xmin,ymin),(xmax,ymax)]
        self.polygons.append(points)

    # edit the filename(data augmentation)
    def editFilename(self, filename):
        fl_name = self.root.find('filename')
        fl_name.text = filename

    # convert polygon to bndbox(pascal voc xml)
    def convertPolygon2BndBox(self, polygon):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in polygon:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)
        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1
        if ymin < 1:
            ymin = 1
        if xmax > self.image_width - 1:
            xmax = self.image_width - 1
        if ymax > self.image_height - 1:
            ymax = self.image_height - 1
        return (int(xmin), int(ymin), int(xmax), int(ymax))

    def editPolygons(self, polygons):
        self.polygons = polygons

    def savePascalVocXML(self,targetFile=None):
        pascal_voc_tree = copy.deepcopy(self.root)
        num = 0

        for object_iter in pascal_voc_tree.findall('object'):
            pts = self.polygons[num]
            valid_pts = []
            for pt in pts[:4]:
                x = max(0, pt[0])
                x = min(x, self.image_width - 1)

                y = max(0, pt[1])
                y = min(y, self.image_height - 1)
                valid_pts.append([x,y])

            # print(len(valid_pts))
            # todo:若box超过一半在图片外则舍弃
            # area = abs((pts[4][0] - pts[5][0]) * (pts[4][1] - pts[5][1]))
            # area_inside = abs((valid_pts[4][0] - valid_pts[5][0]) * (valid_pts[4][1] - valid_pts[5][1]))
            # if area_inside * 1.0 / area < 0.5:
            #     continue

            #增强后点的顺序以及本身的xmin,ymin,xmax,ymax关系已经改变，重新确定
            valid_pts = np.array(valid_pts)
            sort_index = valid_pts[:, 0].argsort()
            valid_pts = valid_pts[sort_index]
            bndbox = object_iter.find("bndbox")

            leftpts = valid_pts[[0,1],:]
            rtpts = valid_pts[[2,3],:]
            sort_index = leftpts[:,1].argsort()
            leftpts = leftpts[sort_index]
            sort_index = rtpts[:, 1].argsort()
            rtpts = rtpts[sort_index]

            x1 = bndbox.find('x1')
            x1.text = str(leftpts[0][0])
            y1 = bndbox.find('y1')
            y1.text = str(leftpts[0][1])

            x2 = bndbox.find('x2')
            x2.text = str(rtpts[0][0])
            y2 = bndbox.find('y2')
            y2.text = str(rtpts[0][1])

            x3 = bndbox.find('x3')
            x3.text = str(rtpts[1][0])
            y3 = bndbox.find('y3')
            y3.text = str(rtpts[1][1])

            x4 = bndbox.find('x4')
            x4.text = str(leftpts[1][0])
            y4 = bndbox.find('y4')
            y4.text = str(leftpts[1][1])

            xmin = bndbox.find('xmin')
            xmin.text = str(np.min(valid_pts[:,0]))
            ymin = bndbox.find('ymin')
            ymin.text = str(np.min(valid_pts[:,1]))

            xmax = bndbox.find('xmax')
            xmax.text = str(np.max(valid_pts[:,0]))
            ymax = bndbox.find('ymax')
            ymax.text = str(np.max(valid_pts[:,1]))
            num = num + 1

        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')
        prettifyResult = self.prettify(pascal_voc_tree)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


if __name__ == '__main__':
    voc_dir = "/home/wz/Data/VIN/drving_license_cut/Seal_Regression/"
    xml_dir = os.path.join(voc_dir, 'xml_aug')
    img_dir = os.path.join(voc_dir, 'img_aug')
    xmls = os.listdir(xml_dir)
    with open('/home/wz/Data/VIN/drving_license_cut/Seal_Regression/train.txt', 'w') as outfile:
        for xml_ in xmls:
            img_name = xml_[0:-4] + '.jpg'
            if not os.path.exists(os.path.join(img_dir, img_name)):
                continue

            reader = TextboxReader(os.path.join(xml_dir, xml_))
            polygons = reader.getPolygons()
            height = reader.image_height
            width = reader.image_width

            lefttop = polygons[0][0]
            rttop = polygons[0][1]
            rtbt = polygons[0][2]
            leftbt = polygons[0][3]

            # left_top_t_x = (lefttop[0] - width / 2) * 1.0/ width
            # left_top_t_y = (lefttop[1] - height / 2) * 1.0 / height
            # rt_top_t_x = (rttop[0] - width / 2) * 1.0 / width
            # rt_top_t_y = (rttop[1] - height / 2) * 1.0 / height
            #
            # rtbt_t_x = (rtbt[0] - width / 2) * 1.0 / width
            # rtbt_t_y = (rtbt[1] - height / 2) * 1.0 / height
            # leftbt_t_x = (leftbt[0] - width / 2) * 1.0 / width
            # leftbt_t_y = (leftbt[1] - height / 2) * 1.0/ height

            left_top_t_x = lefttop[0]  / (width * 224.0)
            left_top_t_y = lefttop[1]  / (height * 224.0)
            rt_top_t_x = (rttop[0])  / (width * 224.0)
            rt_top_t_y = (rttop[1]) / (height * 224.0)

            rtbt_t_x = (rtbt[0]) / (width * 224.0)
            rtbt_t_y = (rtbt[1]) / (height * 224.0)
            leftbt_t_x = (leftbt[0]) / (width * 224.0)
            leftbt_t_y = (leftbt[1]) / (height * 224.0)

            outfile.writelines(img_name
                               + ' ' + str(left_top_t_x)+' '+str(left_top_t_y)
                               + ' ' + str(rt_top_t_x) + ' ' + str(rt_top_t_y)
                               + ' ' + str(rtbt_t_x) + ' ' + str(rtbt_t_y)
                               + ' ' + str(leftbt_t_x) + ' ' + str(leftbt_t_y)
                               + '\n')
            print('Write to file %s successfully!' % (img_name))
    outfile.close()



