# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 11:19:50 2014
readfile:
read the information to get the train or test pairs in the txt
@author: mountain
"""

import linecache
import random


class read_file(object):
    def __init__(self, pth1, num=None):
        self.pth1 = pth1
        self.num = num
        linecache.clearcache()
        self.total = int(linecache.getline(pth1, 1))
        if num == None:
            self.num = self.total

    def person_pair(self):
        person = []
        tmp = range(2, self.total + 2)

        # process part of
        if self.num != self.total:
            # line_label: save the labels of the line to be processed
            line_label = random.sample(tmp, self.num)

            for i in range(len(line_label)):
                person.append(self.__Extrct_flnm(line_label[i]))
        # process all
        else:
            for i in tmp:
                person.append(self.__Extrct_flnm(i))

        return person

    def person_mispair(self):
        person = []
        tmp = range(self.total + 2, self.total * 2 + 2)

        if self.num != self.total:
            # line_label: save the labels of the line to be processed
            line_label = random.sample(tmp, self.num)

            for i in range(len(line_label)):
                person.append(self.__Extrct_flnm(line_label[i]))
        else:
            for i in tmp:
                person.append(self.__Extrct_flnm(i))

        return person

    def __Extrct_flnm(self, line_label):
        tmp = linecache.getline(self.pth1, line_label)
        tmp = tmp.split()

        if len(tmp) == 3:
            flag = 1
            fl_nm1 = tmp[0] + '_' + '0' * (4 - len(tmp[1])) + tmp[1] + '.txt'
            fl_nm2 = tmp[0] + '_' + '0' * (4 - len(tmp[2])) + tmp[2] + '.txt'
        else:
            flag = -1
            fl_nm1 = tmp[0] + '_' + '0' * (4 - len(tmp[1])) + tmp[1] + '.txt'
            fl_nm2 = tmp[2] + '_' + '0' * (4 - len(tmp[3])) + tmp[3] + '.txt'

        pair_info = [fl_nm1, fl_nm2, flag]

        return pair_info
        
        
    

    
    
