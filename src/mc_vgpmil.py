#-*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:16:29 2022
@author: Luz García
"""

from __future__ import print_function
import numpy as np
import cv2
from time import time
from helperfunctions import sigmoid, lambda_fun
#import tensorflow as tf
#import keras
#import gpflow
#import math
#from sklearn.metrics import f1_score, log_loss
#from helperfunctions import RBF
#from scipy.io import savemat

class mc_vgpmil(object):
    def __init__(self, kernel, num_inducing, max_iter, normalize, verbose):
        """
        :param kernel: Specify the kernel to be used
        :param num_inducing: nr of inducing points
        :param max_iter: maximum number of iterations
        :param normalize: normalizes the data before training
        :param verbose: regulate verbosity
        """
        self.kernel = kernel
        self.num_ind = num_inducing
        self.max_iter = max_iter
        self.normalize = normalize
        self.verbose = verbose
        self.lH = np.log(1e12)
        self.lH1 = np.log(1e12 + 1)

    def initialize(self, Xtrain, InstBagLabel, Bags, MultiBagLabel, Z=None, pi=None, mask=None):
        """
        Initialize the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:  n-dim vector with the binary bag label of each instance ; InstBagLabel = bags_binary_class
        :param Bags: n-dim vector with the bag index of each instance; Bags = bags_id
        :param MultiBagLabel: n-dim vector with the class bag label of each instance; MultibagLabel=bag_class
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) n-dim vector to specify instance labels for semi-supervised learning
        :param mask: (opt) n-dim boolean vector to fix instance labels and prevent them from being updated
        """

        self.Ntot = len(Bags)#number of instances

        self.Nbag = 9#number of instances in a bag, needs to be generalized

        self.Nclass = len(np.unique(MultiBagLabel))#number of classes
        self.C = len(np.unique(MultiBagLabel))-1#number of pathological classes
        self.B = len(np.unique(Bags))#number of bags
        self.InstBagLabel = InstBagLabel #bag binary label (max(y_b) in mathematical formulation)
        self.MultiBagLabel = np.zeros((self.Ntot, self.Nclass))
        self.MultiBagLabel[np.arange(MultiBagLabel.size), MultiBagLabel.astype(int)] = 1
        self.MultiBagLabel = self.MultiBagLabel.astype(int)
        self.Bags = Bags

        if self.normalize:
           self.data_mean, self.data_std = np.mean(Xtrain, 0), np.std(Xtrain, 0)
           self.data_std[self.data_std == 0] = 1.0
           Xtrain = (Xtrain - self.data_mean) / self.data_std

        # Initialize Inducing points if not provided, in a non-supervised manner
        if Z is not None:
            assert self.num_ind == Z.shape[0]
            self.Z = Z
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, Z = cv2.kmeans(np.float32(Xtrain), self.num_ind, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
            if self.verbose:
                print("Inducing points are computed")
        self.Z = Z

        # Computing kernel parameters
        self.Kzzi = np.linalg.inv(self.kernel.compute(self.Z) + np.identity(self.num_ind) * 1e-6)
        self.Kzx = self.kernel.compute(self.Z, Xtrain)
        self.KzziKzx = np.dot(self.Kzzi, self.Kzx)
        self.Kxz = self.kernel.compute(Xtrain, self.Z)
        self.KxzKzzi = np.dot(self.Kxz, self.Kzzi)
        self.f_var = 1 - np.einsum("ji,ji->i", self.Kzx, self.KzziKzx)
        self.Ntot_b = [np.sum(self.Bags == b) for b in np.unique(self.Bags)]

        # Initializing parameters for q(u)
        self.m = np.random.randn(self.num_ind, self.C)#(M x C)
        self.S = np.identity(self.num_ind) + np.random.randn(self.C, self.num_ind, self.num_ind) * 0.01 # (C x M x M)

        # Initializing parameters for q(y)
        if pi is not None:
            assert mask is not None, "Don't forget to provide a mask"
            self.mask = mask.copy()
            self.pi = pi.copy()
        else:
            self.pi = np.random.uniform(0, 0.1, size=self.Ntot)
            self.mask = np.ones(self.Ntot) == 1

        # Initializing inference parameters \xi_b^c \gamma \alpha_b
        self.gamma = np.random.randn(self.Ntot)
        self.xi = np.random.randn(self.B, self.C)#B x C
        self.alpha = np.random.rand(self.B)
        self.Lambda_n = lambda_fun(self.gamma)#diagonal values of lambda(\zeta_i), i=1...n
        self.Lambda_k = lambda_fun(self.xi)#diagonal values of lambda(\xi_i), i=1...k

    def q_u_inference(self):#inference of q(U)
        ###################
        ## Calculate E_c ##
        ###################
        # second term
        second = np.sum([self.Lambda_n[n] * self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :]) for n in range(self.Ntot)], axis=0)
        # third term
        third = [np.max(self.pi[self.Bags == b]) for b in np.unique(self.Bags)]
        # fourth term
        for c in range(self.C):
            sth_b = []
            for b in np.unique(self.Bags).astype(int):
                mask = np.where(self.Bags == b)  # instances belonging to the bag
                sth = np.sum([self.KzziKzx[:, [n]].dot(self.KxzKzzi[[m], :]) for n in mask[0] for m in mask[0] if m!=n], axis=0)
                sth2 = np.sum([self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :]) for n in mask[0]])
                sth_b.append([(self.Lambda_k[b, c] / (self.Ntot_b[b]**2)) * third[b]*(sth + sth2)])
            fourth = np.sum(sth_b, axis=0)

            E_c = -0.5 * (self.Kzzi - 2*second - 2*fourth)
            self.S[c, :, :] = -2 * E_c

        ###################
        ## Calculate D_c ##
        ###################
        term_1 = np.sum([self.pi[n] * (self.KxzKzzi[[n], :]) for n in range(self.Ntot)], axis=0)
        term_0 = np.sum([self.m[:, [j]].T for j in range(self.C)], axis=0)
        term_2 = np.sum([self.Lambda_n[n] * term_0.dot(self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :])) for n in range(self.Ntot)], axis=0)
        for c in range(self.C):
            term = []
            for b in np.unique(self.Bags).astype(int):
                mask = np.where(self.Bags == b)  # instances belonging to the bag
                cst = (self.MultiBagLabel[mask[0][0], c + 1] - 0.5 - 0.5 * self.alpha[b]) * self.Lambda_k[b, c] / self.Ntot_b[b]
                term_t = (np.max(self.pi[self.Bags == b]) * cst) * np.sum([(self.KxzKzzi[[n], :]) for n in mask[0]], axis=0)
                term.append([term_t])
            term_3 = np.sum(term, axis=0)

            D_c = term_1 - term_2 + term_3
            D_c = D_c.squeeze(0)

            self.m[:, [c]] = self.S[c, :, :].dot(D_c.T)
            self.F_mean = self.KxzKzzi.dot(self.m)

        return self.S, self.m

    def q_y_inference(self):
        term_f = np.sum(self.F_mean, axis=1)
        Emax = np.empty(len(self.pi))
        for b in np.unique(self.Bags.astype(int)):#as performed in VGPMIL CODE
            mask = self.Bags == b
            pisub = self.pi[mask]
            m1 = np.argmax(pisub)
            tmp = np.empty(len(pisub))
            tmp.fill(pisub[m1])
            pisub[m1] = -99
            m2 = np.argmax(pisub)
            tmp[m1] = pisub[m2]
            Emax[mask] = tmp
        Emax = np.clip(Emax, 0, 1)
        term_s_c = 1 - Emax

        pi = []
        for b in np.unique(self.Bags.astype(int)):
            mask = self.Bags == b
            term_whatever = np.sum(self.MultiBagLabel[mask, 1:][0] * self.F_mean[mask, :].mean(0), axis=0)
            #E[A_b]
            firsty = np.sum((self.F_mean[mask, :].mean(0) - self.xi[b, :]) * 0.5, axis=0)
            erre_1 = 0
            erre_2 = np.sum(self.f_var[mask], axis=0)
            erre = erre_1 + erre_2

            secondy_1 = (self.F_mean[mask, :].mean(0) - self.alpha[b])**2
            secondy_2 = 1/(self.Ntot_b[b]**2) * (erre - self.xi[b, :]**2)
            secondy_3 = np.log(1 + np.exp(self.xi[b, :]))
            secondy = np.sum(self.Lambda_k[b, :] * (secondy_1 + secondy_2) + secondy_3, axis=0)

            EA_b = self.alpha[b] * (1 - self.C/2) + firsty + secondy

            thirty = self.MultiBagLabel[[b], 0] * (self.lH - self.lH1)
            forty = np.sum(self.MultiBagLabel[b, 1:], axis=0) * self.lH1 * self.C

            a_b = term_whatever - EA_b - thirty + forty
            a_n = term_f[mask] + term_s_c[mask] * a_b
            pi_b = sigmoid(a_n)
            pi = np.concatenate((pi, pi_b), axis=0)

        return pi

    def parameter_inference(self):
        #alpha
        for b in np.unique(self.Bags.astype(int)):
            mask = self.Bags == b
            num = 2 * np.sum(self.Lambda_k[b, :] * (self.F_mean[mask, :].mean(0)), axis=0) - (1-self.C/2)
            den = 2 * np.sum(self.Lambda_k[b, :], axis=0)
            self.alpha[b] = num / den
            for c in range(self.C):

                #xi
                xi_c = np.sum((self.F_mean[mask, c].mean(0) - self.alpha[b])**2, axis=0)
                xi_2_r1 = 0
                xi_2_r2 = np.sum(self.f_var[mask], axis=0)
                xi_2 = xi_2_r1 + xi_2_r2
                xi_bc = 1 / (self.Ntot_b[b]**2) * xi_2 + xi_c
                self.xi[b, c] = xi_bc

        #gamma
        F_mean_2 = [(self.KxzKzzi[n, :].dot(self.m))**2 for n in range(self.Ntot)]
        term_f_2 = np.sum(F_mean_2, axis=1)
        for n in range(self.Ntot):
            term_var = self.C * self.f_var[n]
            term_trace = np.sum(np.trace([self.KzziKzx[:, [n]] * self.KxzKzzi[[n], :] * self.S[c, :, :] for c in range(self.C)]), axis=0)
            term_crossed = np.sum([self.m[:, [c]].T.dot(self.KzziKzx[:, [n]]).dot(self.KxzKzzi[[n], :]).dot(self.m[:, [j]]) for c in range(self.C) for j in range(self.C) if c != j], axis=1)
            term_crossed = np.sum(term_crossed, axis=0)
            g_n = term_f_2[n] + term_crossed + term_trace + term_var
            self.gamma[n] = g_n

        return self.alpha, self.gamma, self.xi

    def train(self, Xtrain, InstBagLabel, Bags, MultiBagLabel, Z=None, pi=None, mask=None, init=True):
        """
        Train the model
        :param Xtrain: nxd array of n instances with d features each
        :param InstBagLabel:n-dim vector with the binary bag label of each instance
        :param Bags: n-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) n-dim vector to specify instance labels (probabilities) for semi-supervised
        learning
        :param mask: (opt) n-dim boolean vector to fix instance labels and prevent them from being updated
        :param init: (opt) whether to initialize before training
        :param Nbags: number of instances per bag
        :return variational distribution inference

        """
        if init:
            start = time()
            self.initialize(Xtrain, InstBagLabel, Bags, MultiBagLabel, Z=Z, pi=pi, mask=mask)
            stop = time()
            if self.verbose:
                print("Initialized. \tMinutes needed:\t", (stop - start) / 60.)

        for it in range(self.max_iter):
            start = time()
            if self.verbose:
                print("Iter %i/%i" % (it + 1, self.max_iter))

            self.S, self.m = self.q_u_inference()
            self.pi = self.q_y_inference()
            self.alpha, self.gamma, self.xi = self.parameter_inference()

        if self.verbose:
            print("Minutes needed: ", (stop - start) / 60.)


        print('SOME PRELIMINARY METRICS')
        #efmin = np.argmax([self.F_mean[n, :] for n in range(self.Ntot)])
        #MultiBag = np.zeros((self.Ntot, 1))
        #for n in range(self.Ntot):
        #    MultiBag[self.MultiBagLabel[n, 1] == 1] = 0
        #    MultiBag[self.MultiBagLabel[n, 2] == 1] = 1
        #    MultiBag[self.MultiBagLabel[n, 3] == 1] = 2
        #n_0 = len(MultiBag[MultiBag == 0])
        #n_1 = len(MultiBag[MultiBag[n] == 1])
        #n_2 = len(MultiBag[MultiBag[n] == 2])

    def predict(self, Xtest):
        """
        #Predict instances and bag labels

        :param Xtest: mxd matrix of n instances with d features
        :param self: object mc_vgpmil
        :return: instance and bag laebels predictions

        """
        # binary classification (healthy/pathological) prediction of instances (patches of the image) per each of the classes of the problem
        if self.normalize:
            Xtest = (Xtest - self.data_mean) / self.data_std

        Kzx = self.kernel.compute(self.Z, Xtest)
        KzziKzx = np.dot(self.Kzzi, Kzx)

        return sigmoid(np.dot(KzziKzx.T, self.m))