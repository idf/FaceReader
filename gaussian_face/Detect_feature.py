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
dst_path = 'E:\\GPforFR\\data\\lfw_feature1'

# Traversing files
for root, dirs, files in os.walk(dir_path):
    if files:
        # get the image name
        img_name = root.split('\\')[-1]

        for f in files:
            image = cv2.imread(root + '\\' + f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ftr_name = os.path.join(dst_path, f.split('.')[0] + '.txt')
            # if not os.path.exists(ftr_name):

            feature = Mulscl_lbp_feature(image, radius, nei, scale, scale_step, winsize, stride)

            output = open(ftr_name, 'w')
            pickle.dump(feature, output)
            output.close()
    print root