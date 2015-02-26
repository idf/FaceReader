# -*- coding: utf-8 -*-
"""
Created on Thu Jun 05 11:01:09 2014
the main function for Face Recognition
@author: mountain
"""

from Join_Ftr import *
from Read_file import read_file
# from GS_Face import GS_Face
# import numpy.matlib as Nmat
# import numpy.linalg as LA

# the path of the file which saves the Multi-LBP feature
feature_pth = 'E:\\GPforFR\\data\\lfw_feature1'

# the path of the file which saves the txt describing the Target and source domain set
instruc_pth_t = 'E:\\GPforFR\\data\\lfw_view1\\pairsDevTrain.txt'  # http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt
instruc_pth_s = 'E:\\GPforFR\\data\\lfw_view1\\pairsDevTest.txt'  # http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt

# the number of the match pair and the mis-match pair
num = 5

# obtain the pair information
rd_fl = read_file(instruc_pth_t, num)
X1 = rd_fl.person_pair() + rd_fl.person_mispair()

rd_fl = read_file(instruc_pth_s, num)
X2 = rd_fl.person_pair() + rd_fl.person_mispair()

# Construct a Gaussian Face feature class
tGsFtr = Join_Ftr()

# obtain the target-domain joint feature Xtar, and Ytar
Xtar, Ytar = tGsFtr.Constrct_XY(feature_pth, X1)
# Xsrc,Ysrc=tGsFtr.Constrct_XY(feature_pth,X2)

Xt_in, Yt_in = tGsFtr.XY_in(Xtar, Ytar)
# Xs_in,Ys_in=tGsFtr.XY_in(Xsrc,Ysrc)



# gsface=GS_Face(Xt_in,Xs_in)
# delta=
# beta=
# lmodel=gsface.Gs_model(delta,beta,N_ps,N_ns,N_pt,N_nt,q=0)

