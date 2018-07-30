# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#解决python2下的unicode编码问题

import codecs
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import numpy as np

XML_EXT = '.xml'

class TextboxWriter:
    def __init__(self, foldername, filename, imgsize):
        self.foldername = foldername
        self.filename = filename
        self.imgsize = imgsize
        self.boxlist = []
        self.verified = False
        self.XML_EXT = '.xml'

    def set_foldername(self, foldername):
        self.foldername = foldername

    def set_filename(self,filename):
        self.filename = filename

    def set_size(self, imgsize):
        self.imgsize = imgsize

    def clear_bndbox(self):
        '''
        clear last cache for a new image with empty boxlist.
        :return: .
        '''
        self.boxlist = []

    def add_bndbox(self,x1,y1,x2,y2,x3,y3,x4,y4, xmin, ymin, xmax, ymax,
                  name = 'text',content = '###', diffifult = '0'):
        '''
        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param x3:
        :param y3:
        :param x4:
        :param y4:
        :param xmin:
        :param ymin:
        :param xmax:
        :param ymax:
        :param name:
        :param content:
        :param diffifult:
        :return:
        '''
        bndbox = {'x1':x1,'y1':y1,
                  'x2':x2,'y2':y2,
                  'x3':x3,'y3':y3,
                  'x4':x4,'y4':y4,
                  'xmin': xmin, 'ymin': ymin,
                  'xmax': xmax, 'ymax': ymax}
        bndbox['content'] = content
        bndbox['difficult'] = diffifult
        bndbox['name'] = name
        self.boxlist.append(bndbox)

    def save(self, targetFile=None):
        root = self._genXML()
        self._append_objs(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + self.XML_EXT, 'w', encoding='utf-8')
        else:
            out_file = codecs.open(targetFile, 'w', encoding='utf-8')

        prettifyResult = self._prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()

    # private funcs.
    def _genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgsize is None:
            return None

        top = Element('annotation')
        top.set('verified', 'yes' if self.verified else 'no')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgsize[1])
        height.text = str(self.imgsize[0])

        if len(self.imgsize) == 3:
            depth.text = str(self.imgsize[2])
        else:
            depth.text = '1'

        return top

    def _append_objs(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')

            difficult = SubElement(object_item, 'difficult')
            difficult.text = each_object['difficult']
            content = SubElement(object_item, 'content')
            content.text=each_object['content']
            name = SubElement(object_item, 'name')
            name.text = each_object['name']

            bndbox = SubElement(object_item, 'bndbox')
            x1 = SubElement(bndbox, 'x1')
            x1.text = str(each_object['x1'])
            y1 = SubElement(bndbox, 'y1')
            y1.text = str(each_object['y1'])
            x2 = SubElement(bndbox, 'x2')
            x2.text = str(each_object['x2'])
            y2 = SubElement(bndbox, 'y2')
            y2.text = str(each_object['y2'])
            x3 = SubElement(bndbox, 'x3')
            x3.text = str(each_object['x3'])
            y3 = SubElement(bndbox, 'y3')
            y3.text = str(each_object['y3'])
            x4 = SubElement(bndbox, 'x4')
            x4.text = str(each_object['x4'])
            y4 = SubElement(bndbox, 'y4')
            y4.text = str(each_object['y4'])

            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def _prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)


    def _cvt_poly_to_bndbox(self, polygon):
        '''
        convert polygon to bndbox(pascal voc xml)
        :param polygon:
        :return:
        '''
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
        if xmax > self.imgsize[1] - 1:
            xmax = self.imgsize[1] - 1
        if ymax > self.imgsize[0] - 1:
            ymax = self.imgsize[0] - 1
        return (int(xmin), int(ymin), int(xmax), int(ymax))


class TextboxReader:
    def __init__(self, filepath):
        self.objects = []
        self.filepath = filepath
        self.verified = False
        self.image_width = 0
        self.image_height = 0
        self._parseXML()


    # get polygons
    def get_objects(self):
        '''
          get objects list. each item with format of
         '[label, (x1,y1)...(x4,y4),(xmin,ymin), (xmax,ymax)]'
        :return:
        '''
        return self.objects

    def _prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True)

    # parse XML to get polygons
    def _parseXML(self):
        assert self.filepath.endswith('.xml'), "Unsupport file format"
        parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        self.root = xmltree
        filename = xmltree.find('filename').text

        size = xmltree.find('size')
        self.image_width = int(size.find('width').text)
        self.image_height = int(size.find('height').text)
        # print('image height, image width:', self.image_height, self.image_width)

        for object_iter in xmltree.findall('object'):
            label = object_iter.find('name').text
            bndbox = object_iter.find('bndbox')
            self._add_obj(bndbox, label)
        return True

    # from xml 'bndbox' obj get polygon and append it in polygons
    def _add_obj(self, polygon, label):
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
        obj = [label,(x1,y1),(x2,y2),(x3,y3),(x4,y4),(xmin,ymin),(xmax,ymax)]
        self.objects.append(obj)


