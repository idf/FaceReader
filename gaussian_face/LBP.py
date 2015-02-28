# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:38:32 2014

Detect the multiscale lbp feature

Mulscl_lbp_feature(image,radius,neighbors,scale,scale_step,winsize,stride)
    detect the multi scale lbp histogram feature of the image
                  
lbp(image,radius,neighbors,mapping)
    returns the LBP feature of an image
    
getmapping(samples):
    returns a mapping table for LBP u2
    
MulScl_image(image,scale,scale_step):
    produce the image pyramid 

get_Hist(patch,maxvalue):
    get the hist of the patch, the the value from 0 to maxvalue
    
rotateLeft(i,samples):
    rotate left, equal to bitset(bitshift(i,1,samples),1,bitget(i,samples)) 
    
NumberOfSetBits(i)
    equal to sum(bitget(bitxor(i,j),1:samples))
    
@author: mountain
"""

import numpy as np
import cv2


def multi_scale_lbp_feature(image, radius, neighbors, scale, scale_step, winsize, stride):
    """
    detect the multi scale lbp feature of the image,
    with a patch size (2*winsize+1)*(2*winsize+1)
    scale: the total layer of the image pyramid
    scale_step: the scaling between the adjacent layer
    stride: the stride of the  sliding window
    radius: the radius of the circle
    neighbors: the sample numbers in a circle

    feature: return an array, row:the number of the patch
                             col:lbp feature dim*scale

    eg:
      feature=Mulscl_lbp_feature(image,1,8,4,1.25,25,2)
    """

    # get the image pyramid
    img_pyrmd = multi_scale_image(image, scale, scale_step)

    # a mapping table for LBP u2 with 'neighbors' neighbors
    # mapping=getmapping(neighbors)


    # the mapping table for LBP u2-8
    mapping = np.array([0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, \
                        11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, \
                        16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, \
                        17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, \
                        22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, \
                        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, \
                        23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, \
                        24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, \
                        29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, \
                        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, \
                        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, \
                        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, \
                        36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, \
                        58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, \
                        42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, \
                        47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57])


    # calculate the lbp coding for each scale
    lbp_pyrmd = {}

    for i in range(scale):
        # get the image at scale i
        img_scl = img_pyrmd[i]

        # get the lbp coding for the image at scale i
        lbp_pyrmd[i], maxmapping = lbp(img_scl, radius, neighbors, mapping)


    # calculate the multi-scale lbp feature for the image


    ysize = np.size(image, 0)
    xsize = np.size(image, 1)

    # in order to find the correspoing positions in the lbp_feature image at
    # the biggest scale, the begin position in the lbp_feature image at scale 0
    # is define as follows.  + radius: for the lbp feature image is small than the origen
    # image, radius here is integer
    dsize = int(np.ceil(winsize + radius) * (scale_step ** (scale - 1)))

    # the begin and end positions of the sliding window in the lbp feature at scale 0
    # the row and column is begin from 0 here
    min_y = int(dsize)
    min_x = int(min_y)
    max_y = int(np.floor((ysize - dsize - 1 - min_y) / (stride + 1)) * (stride + 1) + min_y)
    max_x = int(np.floor((xsize - dsize - 1 - min_x) / (stride + 1)) * (stride + 1) + min_x)

    # print max_x
    # print max_y
    # print min_x

    # the number of the patches
    patch_num = int(((max_y - min_y) / (stride + 1) + 1) * ((max_x - min_x) / (stride + 1) + 1))
    # print patch_num

    # the multi scale lbp feature
    # the patch slide from the left to the right, up to down
    Mulscl_lbp = np.zeros([patch_num, scale * (maxmapping + 1)])

    for s in range(scale):

        # the lbp feature at scale s
        lbp_current = lbp_pyrmd[s]

        # the number of the patch
        num = 0

        #the column in Mulscl_lbp
        col = np.arange(int(s * (maxmapping + 1)), int((s + 1) * (maxmapping + 1)))

        for i in range(min_y, max_y + 1, stride + 1):
            i_current = int(round(i / (scale_step ** s)))

            for j in range(min_x, max_x + 1, stride + 1):
                j_current = int(round(j / (scale_step ** s)))

                #the current patch
                patch = np.copy(lbp_current[i_current - winsize:i_current + winsize + 1, \
                                j_current - winsize:j_current + winsize + 1])

                Mulscl_lbp[num, col] = get_Hist(patch, maxmapping)

                num = num + 1

    return Mulscl_lbp


def lbp(image, radius, neighbors, mapping):
    """
    LBP returns the LBP feature of an image.
    J = LBP(I,R,N,MAPPING,MODE) returns either a local binary pattern
    coded image or the local binary pattern histogram of an intensity
    image I. The LBP codes are computed using N sampling points on a
    circle of radius R and using mapping table defined by MAPPING.
    """
    d_image = np.copy(image)
    d_image = d_image + 0.0

    # Angle step
    agl = 2 * np.pi / neighbors

    # coordinates of neighbors
    spoints = np.zeros([neighbors, 2])

    for i in range(neighbors):
        spoints[i, 0] = -radius * np.sin(i * agl)
        spoints[i, 1] = radius * np.cos(i * agl)

    # determine the dimensions of the input image
    ysize, xsize = np.shape(image)

    miny = np.min(spoints[:, 0])
    maxy = np.max(spoints[:, 0])
    minx = np.min(spoints[:, 1])
    maxx = np.max(spoints[:, 1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey = np.ceil(max(maxy, 0.0)) - np.floor(min(miny, 0.0)) + 1
    bsizex = np.ceil(max(maxx, 0.0)) - np.floor(min(minx, 0.0)) + 1

    # Coordinates of origin (0,0) in the block
    origy = 1 - np.floor(min(miny, 0.0))
    origx = 1 - np.floor(min(minx, 0.0))

    if xsize < bsizex or ysize < bsizey:
        print "The size of the image is smaller than the size of the patch"

    # Calculate dx and dy
    dx = xsize - bsizex
    dy = ysize - bsizey

    # Fill the center pixel matrix C
    C = np.copy(image[origy - 1:origy + dy, origx - 1:origx + dx])
    d_C = C + 0.0


    #Initialize the result matrix with zeros.
    result = np.zeros([dy + 1, dx + 1])

    #Compute the LBP code image

    for i in range(neighbors):
        y = spoints[i, 0] + origy
        x = spoints[i, 1] + origx

        fy = np.floor(y)
        cy = np.ceil(y)
        ry = round(y)
        fx = np.floor(x)
        cx = np.ceil(x)
        rx = round(x)

        #check the interpolation needed or not
        if (abs(x - rx) < 1e-06) and (abs(y - ry) < 1e-6):
            #interpolation is not needed
            N = np.copy(image[ry - 1:ry + dy, rx - 1:rx + dx])
            D = N >= C
        else:
            #interpolation is needed
            ty = y - fy
            tx = x - fx

            #Calculate the interpolation weights of the four neighbors
            w1 = (1 - tx) * (1 - ty)
            w2 = tx * (1 - ty)
            w3 = (1 - tx) * ty
            w4 = tx * ty

            #Compute interpolated pixel values
            N = w1 * d_image[fy - 1:fy + dy, fx - 1:fx + dx] + w2 * d_image[fy - 1:fy + dy, cx - 1:cx + dx] + \
                w3 * d_image[cy - 1:cy + dy, fx - 1:fx + dx] + w4 * d_image[cy - 1:cy + dy, cx - 1:cx + dx]
            D = N >= d_C

        #update the result

        v = 2 ** i
        result = result + v * D

    #mapping        
    #bins = np.zeros(np.max(mapping)+1)
    for i in range(np.size(result, 0)):
        for j in range(np.size(result, 1)):
            lbpvalue = mapping[int(result[i, j])]
            result[i, j] = lbpvalue
            #the histogram of LBP feature of the whole image           
            #bins[lbpvalue] = bins[lbpvalue]+1

    '''
    if mode=='nh':
        bins=bins/np.sum(bins)
    '''

    result_full = np.zeros([ysize, xsize])
    result_full[origy - 1:origy + dy, origx - 1:origx + dx] = result

    return result_full, np.max(mapping)


def getmapping(samples):
    # returns a mapping table for LBP u2
    # samples: the number of the neighbors adopted in LBP

    mapping = range(2 ** samples)

    # numbers of patterns in LBP-u2
    newMax = samples * (samples - 1) + 3

    index = 0

    for i in range(2 ** samples):
        # rotate left
        j = rotateLeft(i, samples)

        # number of 1->0 and 0->1 transitions in binary string x is equal to the /
        # number of 1-bits in XOR(x,Rotate left(x))
        # numt = sum(bitget(bitxor(i,j),1:samples))
        numt = NumberOfSetBits(i ^ j)

        if numt <= 2:
            mapping[i] = index
            index = index + 1
        else:
            mapping[i] = newMax - 1
    return mapping


def rotateLeft(i, samples):
    # j = bitset(bitshift(i,1,samples),1,bitget(i,samples)), rotate left
    bg = ((i & (1 << (samples - 1))) >> (samples - 1))
    bs = (i << 1) & (int(2 ** samples) - 1)
    j = (bs + bg) & (int(2 ** samples) - 1)
    return j


def NumberOfSetBits(i):
    # number of 1->0 and 0->1 transitions in binary string x is equal to the /
    # number of 1-bits in XOR(x,Rotate left(x))
    # numt = sum(bitget(bitxor(i,j),1:samples))
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def multi_scale_image(image, scale, scale_step):
    # produce the image pyramid
    # image: the original image
    # scale: the scale of the image pyramid
    # scale_step: the scaling

    # the scaling
    # scale_step=1.25

    img_scl = np.copy(image)

    # the image pyramid with different scaling
    img_pyrmd = {}
    img_pyrmd[0] = img_scl

    for i in range(1, scale):
        row_size = round(np.size(img_scl, 0) / scale_step)
        clmn_size = round(np.size(img_scl, 1) / scale_step)
        img_scl = cv2.resize(img_scl, (int(clmn_size), int(row_size)))
        img_pyrmd[i] = np.copy(img_scl)

    return img_pyrmd


def get_Hist(patch, maxvalue):
    # get the hist of the patch, the the value from 0 to maxvalue

    # initial the hist
    hist = np.zeros(maxvalue + 1)

    # get the hist
    for i in range(np.size(patch, 0)):
        for j in range(np.size(patch, 1)):
            v = patch[i, j]
            hist[v] = hist[v] + 1

    return hist