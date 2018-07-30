import os
import shutil

old_image_folder = '/home/wz/Desktop/FuGuiData/topview1126_and_1127_VOC/VOC2007/JPEGImages-old'
train_image_folder = '/home/wz/Desktop/FuGuiData/topview1126_and_1127_VOC/VOC2007/JPEGImages'
merge_out_folder = '/home/wz/Desktop/FuGuiData/topview1126_and_1127_VOC/VOC2007/imagestotest'

full_images = os.listdir(old_image_folder)
train_images = os.listdir(train_image_folder)

index = 0
for name in full_images:
    if not name in train_images:
        shutil.copy(os.path.join(old_image_folder, name), os.path.join(merge_out_folder, name))
        index += 1
print ("Fully merged images: ", index)