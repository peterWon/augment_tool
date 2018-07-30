#!/usr/bin/env python
#-- coding:utf-8 --
import os.path as osp
import os
import cv2
import time
import copy
from pascal_voc_io import  *

cur_dir = osp.dirname(__file__)

#caution,cfg file must be test mode,or it will raise astype error!!!!!!!!!, SB, did you remember it?
cfg_file = '/home/wz/testProjects/driving_license/models/pvanet/test.yml'
net_pt = '/home/wz/testProjects/driving_license/models/pvanet/test.prototxt'
net_weight = '/home/wz/testProjects/driving_license/models/pvanet/driving_license.caffemodel'
result_dir = '/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped'
test_dirs =['/home/wz/testProjects/driving_license/data/pvashape_data']
xml_writing_dirs = ['/home/wz/Data/VIN/drving_license_cut/ROI/PVA_shaped/xml']
image_dirs = test_dirs

CLASSES=['__background__']
with open('/home/wz/testProjects/driving_license/models/pvanet/classes_name.txt', 'r') as labelfile:
    for line in labelfile.readlines():
        CLASSES.append(line[:-1])

print(CLASSES)

def getIOU(Reframe,GTframe):
    ''' Rect = [x1, y1, x2, y2] '''
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0
    else:
        Area = width*height 
        Area1 = width1*height1 
        Area2 = width2*height2
        ratio = Area*1.0/(Area1+Area2-Area)
    return ratio


def delete_box_iou(old_detections,thresh=0.4):
    new_detections = copy.copy(old_detections)
    index = []
    #ioulist = []
    for i in range(len(new_detections)): # 0 -- len-1
        for j in range(i+1,len(new_detections)):
            iou = getIOU(new_detections[i][1:5],new_detections[j][1:5])
            if iou >= thresh :
                #ioulist.append(iou)
                if new_detections[i][5] >= new_detections[j][5]:  
                    index.append(j)
                else:
                    index.append(i)
    output = []
    for idx,detec in enumerate(new_detections):
        flag = 0
        for i in index:
            if idx == i:
                flag=1
        if flag == 0:
            output.append(detec)
    for idx in index:
        new_detections[idx]
    return output


if __name__ == '__main__':
    # set mode
    recognition.set_mode('gpu',0)

    # if ClASSES changed, update the test.prototxt
    recognition.change_test_prototxt(net_pt, len(CLASSES))
    net = recognition.load_net(cfg_file, net_pt, net_weight)
    #trigger the camera
    #image = recognition.take_picture(0)

    # just like the predefined data structure
    results = {}

    for test_dir, xml_writing_dir, image_dir in zip(test_dirs, xml_writing_dirs, image_dirs):
        test_imgs = os.listdir(test_dir)
        for img_path in test_imgs:
            for cls_ind, cls in enumerate(CLASSES[1:]):
                results[cls] = 0
            image = cv2.imread(osp.join(test_dir,img_path))
            print(image.shape)
            timer = Timer()
            timer.tic()
            detections = recognition.detect(net, image, CLASSES, 0.4)
            timer.toc()
            print ('Detection took {:.3f}s').format(timer.total_time)

            new_detections = delete_box_iou(detections, 0.4) # iou thresh
            tmp = PascalVocWriter(image_dir, img_path[:-4], image.shape)
            for detection in new_detections:
                results[detection[0]] = results[detection[0]] + 1
                tmp.addBndBox(int(detection[1]), int(detection[2]), int(detection[3]), int(detection[4]), str(detection[0]))
            tmp.save(xml_writing_dir + "/" + img_path[:-4] + XML_EXT)
            # cv2.rectangle(image,(int(detection[1]),int(detection[2])),(int(detection[3]),int(detection[4])),(255,255,255),2)
            # cv2.putText(image,detection[0]+' '+str(detection[5]),(int(detection[1]),int(detection[2])),0,0.5,(0,0,255),2)
        # print results
        # rst_path = img_path[:-4]+"-res.jpg"
        # cv2.imwrite(osp.join(result_dir, rst_path), image)
