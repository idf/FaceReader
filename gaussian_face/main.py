# -*- coding: utf-8 -*-
"""
Created on Thu Jun 05 11:01:09 2014
the main function for Face Recognition
@author: mountain
"""

from Join_Ftr import *
from ReadFile import ReadFile
from GS_Face import GsFace


# the path of the file which saves the Multi-LBP feature
feature_pth = 'E:\\GPforFR\\data\\lfw_feature5'

# the path of the file which saves the txt describing the Target and source domain set
instruc_pth_t = 'E:\\GPforFR\\data\\lfw_view1\\pairsDevTrain.txt'  # http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt
instruc_pth_s = 'E:\\GPforFR\\data\\lfw_view1\\pairsDevTest.txt'  # http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt

# the number of the match pair and the mis-match pair
num = 5

# obtain the pair information
read_file = ReadFile(instruc_pth_t, num)
X1 = read_file.person_pair() + read_file.person_mispair()

read_file = ReadFile(instruc_pth_s, num)
X2 = read_file.person_pair() + read_file.person_mispair()

# Construct a Gaussian Face feature class
gs_feature = Join_Ftr()

# obtain the target-domain joint feature Xtar, and Ytar
Xtar, Ytar = gs_feature.Constrct_XY(feature_pth, X1)
Xsrc, Ysrc = gs_feature.Constrct_XY(feature_pth, X2)

Xt_in, Yt_in = gs_feature.XY_in(Xtar, Ytar)
Xs_in, Ys_in = gs_feature.XY_in(Xsrc, Ysrc)


gsface = GsFace(Xt_in, Xs_in)
# delta =
# beta =
# How to obtain N_ps, N_ns, N_pt, N_nt
# lmodel = gsface.Gs_model(delta, beta, N_ps, N_ns, N_pt, N_nt, q=0)