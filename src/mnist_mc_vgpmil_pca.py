# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:17:52 2021

@author: Miguel
"""

import keras
import gpflow
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss
from helperfunctions import RBF
from mcv import mc_vgpmil
from bags_preprocesing_basico import create_bags_multiclass, plot_bag
from numpy import savetxt

# !pip install git
# !git clone  https://github.com/manuelhaussmann/vgpmil.git
# from vgpmilhelperfunctions import RBF
# from vgpmilvgpmil import vgpmil

# For reproducibility
np.random.seed(12)
# train_x_matrix: data for train with pca
# train_x_matrix


train_x_matrix, train_x_matrix_orig, test_x_matrix, test_matrix_orig, bags_id_train, bags_id_test, \
bags_class_train, bags_class_test, instances_class_train, \
instances_class_test = create_bags_multiclass('mnist')

"""
With function plot_bag you can check that bags are created correctly. Example:
    bags_class_train shows me that instance 8000 (print bags_class_train[8000]) belongs to bag class 3, and the id
    of the bag is (print bags_id_train[8000]) 770. 
    If I exectute plot_bag(770,train_x_matrix_orig,bags_id_train) I can see a bag with 5 zeros and 4 threes. 

"""
# plot_bag(770,train_x_matrix_orig,bags_id_train)

kernel = RBF()
vgpmil_model = mc_vgpmil(kernel=kernel,
                         num_inducing=int(8),
                         # num_inducing = int(128),
                         max_iter=int(50),
                         normalize=True,
                         verbose=True)

vgpmil_model.train(train_x_matrix, bags_id_train, bags_class_train, Z=None, pi=None, mask=None, init=True)

"""
predictions_y, predictions_C = vgpmil_model.predict(test_x)
y_pred = predictions_y.copy()
#y esto ? implementamos aquí la interpretación de la sigmoide?
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0

# ?????????
#por razones de código. test_y y train_y tenían antes 
#valores de 0 a 1, y no 0 y 1?
train_y[train_y!=1] = 0
test_y[test_y!=1] = 0

print('test', np.mean(test_y==y_pred))
print('f1', f1_score(test_y, y_pred))
print('log loss', log_loss(test_y, predictions))


Log loss, aka logistic loss or cross-entropy loss.
This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, 
defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data 
y_true. The log loss is only defined for two or more labels
"

#si todas las etiquetas de mi bolsas son iguales me da un error en f1 (log_loss) en bag_metrics
bag_results = bag_metrics(vgpmil_model, test_x, T_test, bag_index_test, 'VGPMIL')

print(bag_results)
""""