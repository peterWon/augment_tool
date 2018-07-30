import os
import shutil

imgdir = '/home/wz/Data/SyntheticChineseStringDataset/Images_and_tfrecords/images'
testset_txt = '/home/wz/Data/SyntheticChineseStringDataset/test.txt'
testset_dir = '/home/wz/Data/SyntheticChineseStringDataset/testset'

with open(testset_txt, 'r') as infile:
    lines = infile.readlines()
    for line_ in lines:
        name = line_.split(' ')[0]
        if os.path.exists(os.path.join(imgdir, name)):
            shutil.copy(os.path.join(imgdir, name), os.path.join(testset_dir, name))
            print('Copy file %s successfully!' % name)
        else:
            print('No such file: %s.' % name)