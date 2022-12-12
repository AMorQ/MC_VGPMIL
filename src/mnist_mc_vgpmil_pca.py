# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:17:52 2021

@author: Miguel
"""

#import keras
#import gpflow
#import numpy as np
#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.metrics import f1_score, log_loss
from helperfunctions import RBF
from mcvgpmil2_va import mc_vgpmil
from bags_preprocesing import create_bags_multiclass#, find_bag_elements
#from numpy import savetxt

# For reproducibility 
np.random.seed(12)

#extract database, construct train and test data
train_x_matrix, test_x_matrix, bags_id_train, bags_id_test, bags_binary_class_train,\
        bags_binary_class_test, bags_class_train, bags_class_test, instances_class_train,\
            instances_class_test = create_bags_multiclass('mnist')

kernel = RBF()
vgpmil_model = mc_vgpmil(kernel=kernel,
                      num_inducing=int(50),
                      max_iter=int(10),
                      normalize=True,
                      verbose=True)

#train the model
vgpmil_model.train(train_x_matrix, bags_binary_class_train, bags_id_train, bags_class_train, Z=None, pi=None, mask=None, init=True)

#predict based on the model trained
"""
predictions_y, predictions_C = vgpmil_model.predict(test_x)
y_pred = predictions_y.copy()

print('test', np.mean(test_y==y_pred))
print('f1', f1_score(test_y, y_pred))
print('log loss', log_loss(test_y, predictions))

print(bag_results)
"""