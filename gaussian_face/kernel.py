# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:38:48 2014
kernels:
    K(xi,xj)=alpha*exp(-0.5*sum(gamma_m*(xi_m-xj_m)^2,m=1,2,..d))+deta(xi,xj)/beta
    refered to the work by James Hensman in github https://github.com/jameshensman/pythonGPLVM
@author: mountain
"""

import numpy as np


class Kernel:
    def __init__(self, alpha, gammas):
        self.alpha = np.exp(alpha)
        self.gammas = np.exp(gammas)
        self.dim = gammas.size
        self.nparams = self.dim + 1

    def set_params(self, params):
        assert params.size == self.nparams
        self.alpha = np.exp(params).copy().flatten()[0]
        self.gammas = np.exp(params).copy().flatten()[1:]

    def get_params(self):
        # print np.hstack((self.alpha,self.gammas,self.theta1,self.theta2))
        return np.log(np.hstack((self.alpha, self.gammas)))

    def __call__(self, x1, x2):
        """
        return K
        """
        if x1.size / len(x1) == 1:
            N1 = 1
            D1 = x1.size
        else:
            N1, D1 = x1.shape
        if x2.size / len(x2) == 1:
            N2 = 1
            D2 = x2.size
        else:
            N2, D2 = x2.shape
        assert D1 == D2, "x1 dimension not equal to x2"
        assert D1 == self.dim, "data dimension not equal to the kernel"
        diff = x1.reshape(N1, 1, D1) - x2.reshape(1, N2, D2)
        diff = self.alpha * np.exp(-np.sum(np.square(diff) * self.gammas, -1) / 2)
        # diff=self.alpha*np.exp(-np.sum(np.square(diff)*self.gammas,-1))+self.theta1+np.eye(N1,N2)/self.theta2
        return diff

    def gradients(self, x1):
        """
        the gradients wrt params
        """
        N1, D1 = x1.shape
        diff = x1.reshape(N1, 1, D1) - x1.reshape(1, N1, D1)
        sqdiff = np.sum(np.square(diff) * self.gammas, -1)
        expdiff = np.exp(-sqdiff / 2)
        # expdiff=np.exp(-sqdiff)
        grads = [-0.5 * g * np.square(diff[:, :, i]) * self.alpha * expdiff for i, g in enumerate(self.gammas)]
        # grads=[-g*np.square(diff[:,:,i])*self.alpha*expdiff for i,g in enumerate(self.gammas)]
        grads.insert(0, self.alpha * expdiff)
        return grads

    def gradients_wrt_data(self, x1, indexn=None, indexd=None):
        """
        the derivative matrix with regart to the data
        return a list of D matrix(N*N)
        """
        N1, D1 = x1.shape
        diff = x1.reshape(N1, 1, D1) - x1.reshape(1, N1, D1)
        sqdiff = np.sum(np.square(diff) * self.gammas, -1)
        expdiff = np.exp(-sqdiff / 2)
        # expdiff=np.exp(-sqdiff)

        rslt = []

        if (indexn is None) and (indexd is None):
            for n in range(N1):
                for d in range(D1):
                    K = np.zeros((N1, N1))
                    K[n, :] = -self.alpha * expdiff[n, :] * self.gammas[d] * (x1[n, d] - x1[:, d])
                    # K[n,:]=-2*self.alpha*self.gammas[d]*(x1[n,d]-x1[:,d])*expdiff[n,:]
                    K[:, n] = K[n, :]
                    rslt.append(K.copy())
            return rslt

        else:
            K = np.zeros((N1, N1))
            K[indexn, :] = -self.alpha * self.gammas[indexd] * (x1[indexn, indexd] - x1[:, indexd]) * expdiff[indexn, :]
            K[:, indexn] = K[indexn, :]
            return K.copy()