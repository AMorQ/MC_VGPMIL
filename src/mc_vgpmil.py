# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:16:29 2022
@author: Luz García
"""

from __future__ import print_function
import numpy as np
import cv2
from time import time
from helperfunctions import sigmoid, lambda_fun


# import tensorflow as tf
# import keras
# import gpflow
# import math
# from sklearn.metrics import f1_score, log_loss
# from helperfunctions import RBF
# from scipy.io import savemat

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
        #self.lH1 = np.log(1e12 + 1)

    def initialize(self, Xtrain, Bags, MultiBagLabel, Z=None, pi=None, mask=None):
        """
        Initialize the model
        :param Xtrain: Nxd array of n instances with d features each
        :param Bags: N-dim vector with the bag index of each instance; Bags = bags_id
        :param MultiBagLabel: N-dim vector with the class bag label of each instance; MultibagLabel=bag_class
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) N-dim vector to specify instance labels for semi-supervised learning
        :param mask: (opt) N-dim boolean vector to fix instance labels and prevent them from being updated
        """

        self.Ntot = len(Bags)  # number of instances

        #self.Nbag = 9  # number of instances in a bag, needs to be generalized

        self.Nclass = len(np.unique(MultiBagLabel))  # number of classes
        self.C = len(np.unique(MultiBagLabel)) - 1  # number of pathological classes
        self.B = len(np.unique(Bags))  # number of bags
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
            _, _, Z = cv2.kmeans(np.float32(Xtrain), self.num_ind, None, criteria, attempts=10,
                                 flags=cv2.KMEANS_RANDOM_CENTERS)
            if self.verbose:
                print("Inducing points are computed")
        self.Z = Z
        #print(Xtrain.mean(0), Xtrain.std(0))
        # Computing kernel parameters
        self.Kzzi = np.linalg.inv(self.kernel.compute(self.Z) + np.identity(self.num_ind) * 1e-6)
        self.Kzx = self.kernel.compute(self.Z, Xtrain)
        self.KzziKzx = np.dot(self.Kzzi, self.Kzx)
        self.Kxz = self.kernel.compute(Xtrain, self.Z)
        #self.Kxx = self.kernel.compute(Xtrain, Xtrain)
        self.KxzKzzi = np.dot(self.Kxz, self.Kzzi)
        self.f_variance = np.einsum("ij,jk", self.Kxz, self.KzziKzx) #producto matricial normal
        self.f_var = (1 - np.einsum("ii->i", self.f_variance))*1.5#cojo la diagonal
        # ????????????????????????????
        #self.f_var = 1 - np.einsum("ii->i", self.f_variance)
        #es normal que me de una varianza tan pequeña? me da próxima a cero
        self.Ntot_b = [np.sum(self.Bags == b) for b in np.unique(self.Bags)]

        # Initializing parameters for q(u)
        self.m = np.random.randn(self.num_ind, self.C)  # (M x C)
        self.S = np.identity(self.num_ind) + np.random.randn(self.C, self.num_ind, self.num_ind) * 0.1  # (C x M x M)

        # Initializing parameters for q(y)
        if pi is not None:
            assert mask is not None, "Don't forget to provide a mask"
            self.mask = mask.copy()
            self.pi = pi.copy()
        else:
            self.pi = np.random.uniform(0, 0.5, size=self.Ntot) #inicializo todas
            #las instancias como sanas
            self.mask = np.ones(self.Ntot) == 1

        # Initializing inference parameters \xi_b^c \gamma \alpha_b
        self.gamma = np.random.randn(self.Ntot)
        self.xi = np.random.randn(self.B, self.C)  # B x C
        self.alpha = np.random.rand(self.B)
        self.Lambda_n = lambda_fun(self.gamma)  # diagonal values of lambda(\zeta_i), i=1...n
        self.Lambda_k = lambda_fun(self.xi)  # diagonal values of lambda(\xi_i), i=1...k

    def q_u_inference(self):  # inference of q(U)
        ###################
        ## Calculate E_c ##
        ###################
        # second term
        #second_2 = np.sum([self.Lambda_n[n] * (self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :])) for n in range(self.Ntot)],
                        #axis=0)
        #print(second_2)
        second = np.sum(
            [self.Lambda_n[n] * self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :]) for n in range(self.Ntot)],
            axis=0)
        # third term
        third = [np.max(self.pi[self.Bags == b]) for b in np.unique(self.Bags)]
        # fourth term
        for c in range(self.C):
            sth_b = []
            for b in np.unique(self.Bags).astype(int):
                mask = np.where(self.Bags == b)  # instances belonging to the bag
                #tengo mask como una tuple, mask[0] es lo que yo quiero
                sth = np.sum([self.KzziKzx[:, [n]].dot(self.KxzKzzi[[m], :]) for n in mask[0] for m in mask[0] if m != n],axis=0)
                sth2 = np.sum([self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :]) for n in mask[0]], axis=0)
                sth_b.append([(self.Lambda_k[b, c] / (self.Ntot_b[b] ** 2)) * third[b] * (sth + sth2)])
            fourth = np.sum(sth_b, axis=0)

            E_c = -0.5 * (self.Kzzi + 2 * second - 2 * fourth)
            #estos signos pueden estar mal, lo corregí pero
            #no está pasado al latex
            self.S[c, :, :] = -2 * E_c
            #tiene pinta de que S es muy grande
            #la única forma de que se haga más pequeño es que second and fourth compitan con Kzzi, que no puede cambiar
            #pero es que no puedo hacerlo más pequeño

        ###################
        ## Calculate D_c ##
        ###################
        #term_1_what = np.sum(self.pi * self.KxzKzzi, axis=1)
        #term_1 = term_1_what
        term_1 = np.sum([self.pi[n] * (self.KxzKzzi[[n], :]) for n in range(self.Ntot)], axis=0)
        #term_0 = np.sum([self.m[:, [j]].T for j in range(self.C)], axis=1) = 2x15
        term_0 = np.sum([self.m[:, [j]].T for j in range(self.C)], axis=0) #= 1x15
        #la media tiene valores muy altos (negativos o positivos)
        #para la primera clase y valores muy bajos para la segunda

        term_2 = np.sum(
            [self.Lambda_n[n] * term_0.dot(self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :])) for n in range(self.Ntot)],
            axis=0)
        #term_2_2 = np.sum(
            #[self.Lambda_n[n] * term_0.dot(self.KzziKzx[:, [n]] * (self.KxzKzzi[[n], :])) for n in range(self.Ntot)],
            #axis=1)
        for c in range(self.C):
            term = []
            for b in np.unique(self.Bags).astype(int):
                mask = np.where(self.Bags == b)  # instances belonging to the bag
                #cst = (self.MultiBagLabel[mask[0][0], c + 1] - 0.5 - 0.5 * self.alpha[b]) * (self.Lambda_k[b, c]) / \
                      #(self.Ntot_b[b])
                # ?????? si le pongo np.where a mask no me hace falta que sea mask[0][0]
                cst = (self.MultiBagLabel[mask[0], c + 1] - 0.5 - 0.5 * self.alpha[b]) * (self.Lambda_k[b, c]) / (self.Ntot_b[b])
                #qué debería pasar con cts, en principio, diría que será más alta para patológicas
                #pero la diferencia es infinitesimal

                #term_t_1 = [self.KxzKzzi[[t], :] for t in mask[0]]
                #term_t_2 = np.sum(term_t_1, axis=0)
                #term_t_3 = np.max(self.pi[self.Bags==b])
                #term_t_4 = term_t_3 * term_t_2
                term_t = (np.max(self.pi[mask[0]]) * cst)[0] * np.sum([(self.KxzKzzi[[n], :]) for n in mask[0]], axis=0)
                #se me hace chiquitito en segunda iteración por pi
                term.append([term_t])
            term_3 = np.sum(term, axis=0)


            #D_c = term_1 - term_2[[c], :] + term_3
            D_c = term_1 - term_2 + term_3
            D_c = D_c.squeeze(0)

            self.m[:, [c]] = self.S[c, :, :].dot(D_c.T)
            #es aquí donde tengo el problema
            #tengo que poder hacer más pequeño D_c
            #self.m[:, [c]] = self.S[c, :, :].dot(D_c.T)
            self.F_mean = self.KxzKzzi.dot(self.m)

        return self.S, self.m

    def q_y_inference(self):
        term_f = np.sum(self.F_mean, axis=1)
        #term_f tiene que tener valores bajos para instancias/bolsas negativas
        #pero tengo un probelma en los signos
        #TENGO QUE REVISAR LOS CÁLCULOS
        Emax = np.empty(len(self.pi))
        for b in np.unique(self.Bags.astype(int)):  # as performed in VGPMIL CODE
            mask = self.Bags == b
            #mask = np.where(self.Bags == b)
            pisub = self.pi[mask]
            m1 = np.argmax(pisub)
            tmp = np.empty(len(pisub))
            tmp.fill(pisub[m1])
            pisub[m1] = -9999999999999
            #lo tengo que poner mucho más pequeño, en mi caso pi alcanza valores muy
            #pequeños
            #no me hace falta si luego le hago un clip
            m2 = np.argmax(pisub)
            tmp[m1] = pisub[m2]
            Emax[mask] = tmp
        Emax = np.clip(Emax, 0, 1)
        term_s_c = 1 - Emax

        pi = []
        for b in np.unique(self.Bags.astype(int)):
            mask = self.Bags == b
            term_whatever = np.sum(self.MultiBagLabel[mask, 1:][0] * self.F_mean[mask, :].mean(0), axis=0)
            #term_whatever será 0 para las bolsas sanas
            # E[A_b]


            oo = self.F_mean[mask, :].mean(0)
            #las primeras bolsas son sanas. Primeramente, va a estar parecido
            firsty = np.sum((self.F_mean[mask, :].mean(0) - self.xi[b, :]) * 0.5, axis=0)
            erre_1 = 0
            erre_2 = np.sum(self.f_var[mask], axis=0)
            #la suma de las varianzas de las instancias de la bolsa es super bajo
            erre = erre_1 + erre_2

            secondy_1 = (self.F_mean[mask, :].mean(0) - self.alpha[b]) ** 2
            secondy_2 = 1 / (self.Ntot_b[b] ** 2) * (erre) - self.xi[[b], :] ** 2
            #secondy_2 = 1 / (self.Ntot_b[b] ** 2) * (erre - self.xi[b, :] ** 2)
            secondy_3 = np.log(1 + np.exp(self.xi[[b], :]))
            secondy = np.sum(self.Lambda_k[[b], :] * (secondy_1 + secondy_2) + secondy_3, axis=1)

            EA_b = self.alpha[b] * (1 - self.C / 2) + firsty + secondy

            thirty = self.MultiBagLabel[[b], 0] * (self.lH )
            forty = np.sum(self.MultiBagLabel[b, 1:], axis=0) * self.lH * self.C

            a_b = term_whatever - EA_b - thirty + forty
            a_n = term_f[mask] + term_s_c[mask] * a_b
            #necesito que a_n, a_b sean positivos para las bolsas patológicas
            #term_whatever tiene que tirar por encima de EA_b
            #o term_s_c es chiquitito, si a_b no fuera tan grande y negativo, term_f podría hacer
            #positivo a a_n
            #TENGO QUE REVISAR LOS CÁLCULOS
            pi_b = sigmoid(a_n)
            pi = np.concatenate((pi, pi_b), axis=0)
            pi = np.clip(pi, 0, 1)
        #print(oo)
        return pi



    def parameter_inference(self):
        # alpha
        for b in np.unique(self.Bags.astype(int)):
            mask = self.Bags == b
            num = 2 * np.sum(self.Lambda_k[b, :] * (self.F_mean[mask, :].mean(0)), axis=0) - (1 - self.C / 2)
            den = 2 * np.sum(self.Lambda_k[b, :], axis=0)
            self.alpha[b] = num / den
            for c in range(self.C):
                # xi
                xi_c = np.sum((self.F_mean[mask, c].mean(0) - self.alpha[b]) ** 2, axis=0)
                xi_2_r1 = 0
                xi_2_r2 = np.sum(self.f_var[mask], axis=0)
                xi_2 = xi_2_r1 + xi_2_r2
                xi_bc = 1 / (self.Ntot_b[b] ** 2) * xi_2 + xi_c
                self.xi[b, c] = np.sqrt(xi_bc)

        # gamma
        F_mean_2 = (self.KxzKzzi.dot(self.m)) ** 2
        term_f_2 = np.sum(F_mean_2, axis=1)
        term_var = self.C * self.f_var

        for n in range(self.Ntot):
            crossed = [self.m[:, [c]].T.dot(self.KzziKzx[:, [n]]).dot(self.KxzKzzi[[n], :]).dot(self.m[:, [j]]) for c in
                       range(self.C) for j in range(self.C) if c != j]
            term_trace = np.sum(
                np.trace([self.KzziKzx[:, [n]].dot(self.KxzKzzi[[n], :].dot(self.S[c, :, :])) for c in range(self.C)]),
                axis=0)
            term_crossed = np.sum(crossed)
            g = term_f_2[n] + term_crossed + term_trace + term_var[n]
            self.gamma[n] = np.sqrt(g)

        return self.alpha, self.gamma, self.xi

    def train(self, Xtrain, Bags, MultiBagLabel, Z=None, pi=None, mask=None, init=True):
        """
        Train the model
        :param Xtrain: Nxd array of n instances with d features each
        :param Bags: N-dim vector with the bag index of each instance
        :param Z: (opt) set of precalculated inducing points to be used
        :param pi: (opt) N-dim vector to specify instance labels (probabilities) for semi-supervised
        learning
        :param mask: (opt) N-dim boolean vector to fix instance labels and prevent them from being updated
        :param init: (opt) whether to initialize before training
        :param Nbags: number of instances per bag
        :return variational distribution inference

        """
        if init:
            start = time()
            self.initialize(Xtrain, Bags, MultiBagLabel, Z=Z, pi=pi, mask=mask)
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

    def predict(self, Xtest, bags_id_test):
        """
        #Predict instances and bag labels

        :param Xtest: mxd matrix of n instances with d features
        :param self: object mc_vgpmil
        :return: instance and bag laebels predictions

        """
        # binary classification (healthy/pathological) prediction of instances (patches of the image) per each of the classes of the problem
        # esto lo hacemos ya en preprocessing?
        if self.normalize:
            Xtest = (Xtest - self.data_mean) / self.data_std

        print('Media y desviación de Xtest------->', Xtest.mean(0), Xtest.std(0))

        Kxz = self.kernel.compute(Xtest, self.Z)
        Ntest = np.shape(Xtest)[0]
        Kzx = self.kernel.compute(self.Z, Xtest)
        Kxx = self.kernel.compute(Xtest, Xtest)
        #print(Kxx, Kxz, self.Kxz)
        mu = Kxz.dot(self.Kzzi).dot(self.m)
        #en mu tengo valores muy próximos a zero
        # self.F_mean = self.KxzKzzi.dot(self.m) pero F_mean no tiene valores tan pequeños
        #es que Kxz es muy pequeño
        sigma = np.zeros((Ntest, Ntest, self.C))
        for c in range(self.C):
            sigma[:, :, c] = Kxx - Kxz.dot(self.Kzzi).dot(self.S[c, :, :].dot(self.Kzzi - np.identity(self.num_ind))).dot(Kzx)

        argu = np.sum(mu, axis=1)
        N_bags_test = len(np.unique(bags_id_test))
        pT = np.zeros((N_bags_test, self.Nclass))

        for b in np.unique(bags_id_test).astype(int):
            mask = bags_id_test == b
            ef = (np.exp(mu[mask]).mean(0)) / np.sum(np.exp(mu[mask].mean(0)), axis=0)
            #me dice que en ef, hay una división por zero con 10 it y 27 ptos inductores
            pT[b, 1:] = ef #esto se me va de los axis con el toy_example

            #me salen todas las probabilidades iguales para las clases pat
            #la funcion del kernel es muy pequeña, parece que los valores de entrada van bien
            #el problema aqui es que mu es muy grande:
            #dos cosas, tendremos que volver a normalizar para que sean coherente la etiqueta sana y multiclase
            #segundacosa, estaremos inicializando de la forma mejor?
            pT[b, 0] = np.prod(1 - sigmoid(np.sum(mu[mask], axis=1)))

        return sigmoid(argu), pT
