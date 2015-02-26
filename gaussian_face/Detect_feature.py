# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:00:57 2014
Detect_feature
detect the multi-scale LBP feature of the face images
@author: mountain
"""
import os

from LBP import *
import cv2
import pickle
from util.commons_util.logger_utils.Timer import Timer
from multiprocessing import Pool

# the radius in LBP
radius = 1
# the number of the neighbors in LBP
nei = 8
# the scale in Multi-scale LBP
scale = 4
# the ratio in scaling
scale_step = 1.25
# the patch size for getting hist
winsize = int((25 - 1) / 2)
# the stride between the patch in getting hist
stride = 2
# under this condition, the feature dimension is 816*236

# dir_path: the path of the folder containing the image
dir_path = 'E:\\GPforFR\\data\\lfw_p'

# dst_path: the path of the file saving the feature
dst_path = 'E:\\GPforFR\\data\\lfw_feature'


def extract_feature(args):
    image, ftr_name = args
    timer = Timer()
    out_file = open(ftr_name, 'w')
    timer.start()
    feature = Mulscl_lbp_feature(image, radius, nei, scale, scale_step, winsize, stride)
    pickle.dump(feature, out_file)
    out_file.close()
    print timer.end()

if __name__=="__main__":
    # Traversing files
    p = Pool(5)
    for root, dirs, files in os.walk(dir_path):
        if files:
            # get the image name
            img_name = root.split('\\')[-1]

            load = []
            for f in files:
                image = cv2.imread(root + '\\' + f)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ftr_name = os.path.join(dst_path, f.split('.')[0] + '.txt')
                # if not os.path.exists(ftr_name):
                load.append((image, ftr_name))

            p.map(extract_feature, load)
        print root
