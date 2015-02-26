# -*- coding: utf-8 -*-
"""
Created on Wed May 28 20:15:39 2014
Extract the feature of a pair of faces based on GS_Face


Join_feature:
Construct the joint feature of the image pair
Section 5.2 Fig.1(b)
input:
    pth1, pth2:     the path of image pair feature
output:
    jnt_ftr:        the joint feature of the image pair
    
Constrct_XY:
return the X and Y based on the instruction txt
input:
    feature_pth:    the file path where the feature txt is
    X_mtch,X_mtch:  the file contain the match pair and the mis-match pair
output:
    X:              X is a list contains the joint feature X
    Y:              Y is a list contains whether the face belongs to same person

Gs_ftr:
return the Gaussian face feature for a pair of faces
input:
    hyp_para:       the hyper para obtained by GS model
    X:              the joint feature of the pair of faces
output:
    gs_ftr:         the GS face feature


@author: mountain
"""
import pickle
import numpy as np
import os.path
import numpy.matlib as mat


class Join_Ftr(object):
    def Join_feature(self, fl1, fl2):
        # return the joint feature of a pair of images for all the patches
        # note: without the flipped one

        # read the feature
        ftr1 = pickle.load(fl1)
        ftr2 = pickle.load(fl2)

        # Five Landmarks
        # test,take only five patch
        # eye_left: patch_number 293
        # eye_right: patch_number 309
        # nose: patch_number 565
        # mouth_left: patch_number 751
        # mouth_right: patch_numer 763
        row, col = ftr1.shape
        jnt_ftr = np.empty([5, col * 2])
        for i in range(5):
            jnt_ftr[i] = np.append(ftr1[i], ftr2[i])
        #        for i in range(10):
        #            jnt_ftr[i]=np.append(ftr2[i],ftr1[i])
        '''
        #take all patch
        row,col=ftr1.shape
        jnt_ftr=np.empty(row,col*2)
        for i in range(row):
            jnt_ftr[i]=np.append(ftr1[i],ftr2[i])
        for i in range(row):
            jnt_ftr[i]=np.append(ftr2[i],ftr1[i])
        '''
        return jnt_ftr

    def Constrct_XY(self, feature_pth, X_info):
        """
        :param feature_pth: the file path where the feature txt is
        :param X_info: the file contain the information of pair
        :return: return the X and Y based on the dataset
        X：a list , every element is the joint feature of each image pairs(without flipped)
                   the element in X: M*N, M is the number of the patches
                   N is the dimension of the joint feature
        """


        X = []
        Y = []

        for i in range(len(X_info)):
            pth1 = os.path.join(feature_pth, X_info[i][0])
            pth2 = os.path.join(feature_pth, X_info[i][1])
            if os.path.exists(pth1) and os.path.exists(pth2):
                fl1 = open(pth1, 'r')
                fl2 = open(pth2, 'r')

                X.append(self.Join_feature(fl1, fl2))
                Y.append(X_info[i][2])
                fl1.close()
                fl2.close()

        return X, Y

    def XY_in(self, X, Y):
        """

        :param X: the output of the function Constrct_XY
        :param Y:
        :return: the needed input X_in,Y_in of the GS_FACE model
        """
        assert len(X) == len(Y), "the number of the image pair and their correspoing y is not equal"
        num_of_pair = len(X)
        num_of_patch = X[0].shape[0]
        n_ftr = X[0].shape[1]
        X_in = mat.zeros([num_of_pair * num_of_patch * 2, n_ftr])
        Y_in = mat.zeros([num_of_pair * num_of_patch * 2, 1])
        num = 0
        for i in range(num_of_pair):
            Img_pair = X[i]
            for j in range(num_of_patch):
                X_in[num] = Img_pair[j]
                Y_in[num] = Y[i]
                num = num + 1
                X_in[num, 0:n_ftr / 2] = Img_pair[j][n_ftr / 2:]
                X_in[num, n_ftr / 2:] = Img_pair[j][0:n_ftr / 2]
                Y_in[num] = Y[i]
                num = num + 1

        return X_in, Y_in

                
            
    
        


  
    
