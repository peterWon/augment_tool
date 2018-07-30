#-- coding:utf-8 --
import os
import cv2
import sys
sys.path.append('../')
from scipy import ndimage
from scipy import misc
from imgaug import augmenters as iaa
import random

######################################################################
# 用于增强分类网的数据，按照增强基数aug_base_num，将样本数量不足的类别增强到该数目
# 结果就存在原路径对应文件夹下
######################################################################

cls_sample_root_dir = '/home/wz/DataSets/SuNingFT/IMG'
aug_base_num = 300

seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=(2, 4)),
    iaa.AdditiveGaussianNoise(scale = 25),
    # iaa.Crop(px=(0, 16)),
    iaa.Affine(rotate=(-5, 5), mode ='edge'),
    # iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #            translate_px = {"x": (-10, 10), "y": (-10, 10)}, mode='edge'),
    # iaa.Affine(shear = 20, mode = 'edge'),
    iaa.Multiply((0.5, 1.5))
])

if __name__ == '__main__':
    classes = os.listdir(cls_sample_root_dir)
    cls_num = {}
    for cls in classes:
        sub_cls_dir = os.path.join(cls_sample_root_dir, cls)
        cls_num[cls] = len(os.listdir(sub_cls_dir))
        if cls_num[cls] >= aug_base_num:
            continue
        img_names = os.listdir(sub_cls_dir)
        for i in range(aug_base_num - len(img_names)):
            img_name = random.choice(img_names)
            save_name = img_name.split('.')[0] + '_aug_' + str(i) + '.jpg'
            img_path = os.path.join(sub_cls_dir, img_name)
            if not os.path.exists(img_path):
                continue
            image = ndimage.imread(img_path)
            seq_manner = seq.to_deterministic()
            image_aug = seq_manner.augment_image(image)
            misc.imsave(os.path.join(sub_cls_dir, save_name), image_aug)

        print('Save augmented %s images successfully!' %  cls)
