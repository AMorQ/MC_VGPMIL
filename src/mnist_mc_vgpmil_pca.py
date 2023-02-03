# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:17:52 2021
@author: Miguel
@contributors: Luz, Alba
"""

import keras
import gpflow
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, log_loss
from helperfunctions import RBF
from mc_vgpmil import mc_vgpmil
from mnist_data_preprocesing import create_bags_multiclass, plot_bag
from toy_example import toy_example, very_toy_example
from numpy import savetxt
import mlflow

# 4 reproducibility and traceability
mlruns_folder = 'mlruns'
experiment_name = 'toy_example_experiment'
run_name = 'first_run'

mlflow.set_tracking_uri(mlruns_folder)
experiment_id = mlflow.set_experiment(experiment_name=experiment_name)
mlflow.start_run(experiment_id=experiment_id.experiment_id, run_name=run_name)

print(mlflow.get_artifact_uri())#mlruns/1/fb27b49858c2402fa9db8e33389ef4bb/artifacts
print(mlflow.get_tracking_uri())#mlruns
#tengo un problema de directorios y no consigo abrir la interfaz
#pero se me guardan los archivos en el directorio correcto


#el problema con mlflow es una cuestión de directorios o permisos.
#MODIFICAR LA LÍNEA DE COMANDOS PARA VER SI ES UNA CUESTIÓN DE DIRECTORIOS
#hablar Arne sobre los permisos

#de todas formas, ya puedo guardar parámetros

# !pip install git
# !git clone  https://github.com/manuelhaussmann/vgpmil.git
# from vgpmilhelperfunctions import RBF
# from vgpmilvgpmil import vgpmil

# For reproducibility
np.random.seed(12)
# train_x_matrix: data for train with pca
# train_x_matrix


#train_x_matrix, train_x_matrix_orig, test_x_matrix, test_matrix_orig, bags_id_train, bags_id_test, \
#bags_class_train, bags_class_test, instances_class_train, \
#instances_class_test = create_bags_multiclass('mnist')

#x_train, y_train, bag_label = toy_example(9)
#x_test, y_test, bag_label_test = toy_example(6)

x_train, y_train, bag_label = very_toy_example()
"""
With function plot_bag you can check that bags are created correctly. Example:
    bags_class_train shows me that instance 8000 (print bags_class_train[8000]) belongs to bag class 3, and the id
    of the bag is (print bags_id_train[8000]) 770. 
    If I exectute plot_bag(770,train_x_matrix_orig,bags_id_train) I can see a bag with 5 zeros and 4 threes. 

"""
# plot_bag(770,train_x_matrix_orig,bags_id_train)

kernel = RBF()
#num_inducing = 20
num_inducing = len(x_train)
max_iter = 10
#vgpmil_model = mc_vgpmil(kernel=kernel,
#                         num_inducing=int(num_inducing),
#                         # num_inducing = int(128),
#                         max_iter=int(max_iter),
#                         normalize=True,
#                         verbose=True)



vgpmil_model = mc_vgpmil(kernel=kernel,
                         num_inducing=int(num_inducing),
                         # num_inducing = int(128),
                         max_iter=int(max_iter),
                         normalize=False,
                         verbose=True)

mlflow.log_metric("num_inducing", num_inducing)
mlflow.log_metric("max_it", max_iter)

#Y AHORA VUELVEN A SER IGUALES PARA CLASES, DA IGUAL QUE COMBINACION
#CUANDO SUBO EL NÚMERO DE INDUCING_POINTS ME EMPIEZA A DAR NAN
#ES CURIOSO, PASAN DE SER SUPER PEQUEÑOS (EXP PEQUEÑAS) A SUPER GRANDES (EXP GRANDES)
#PARECE QUE EL EQUILIBRIO PUEDA ESTAR TAMBIÉN CON EL NÚMERO MÁXIMO DE ITERACIONES


#vgpmil_model.train(train_x_matrix, bags_id_train, bags_class_train, Z=None, pi=None, mask=None, init=True)
vgpmil_model.train(x_train, bag_label, bag_label, Z=x_train, pi=None, mask=None, init=True)

predictions_y, predictions_C = vgpmil_model.predict(x_test, bag_label_test)
#predictions_y, predictions_C = vgpmil_model.predict(test_x_matrix, bags_id_test)

y_pred = predictions_y.copy()
C_pred = predictions_C.copy()
bag_bags_class_test = [0] * len(np.unique(bags_id_test))

for i in range(len(np.unique(bags_id_test))):
    C_pred_max = np.argmax(C_pred[i])
    mask_class = bags_class_test[bags_id_test == i][0]
    bag_bags_class_test[i] = mask_class
    if C_pred[i, C_pred_max] < 0.25:
        C_pred[1:] = [0, 0, 0]
        C_pred[0] = 100
    else:
        continue



    #bags_class_test está formado por bolsas con 4 elementos y bolsas
    #con 9 elementos
print('bags_test', np.mean(np.argmax(C_pred, axis=1)==bag_bags_class_test))
#todas las probabilidades sanas son la misma, todas las probabilidades
#patológicas se diferencian muy poco, no está aprendiendo
mino = np.mean(np.argmax(C_pred, axis=1)==bag_bags_class_test)
mlflow.log_metric("bags_test", mino.astype(np.float32))

#y esto ? implementamos aquí la interpretación de la sigmoide?
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
instances_class_train[instances_class_train!=1] = 0
instances_class_test[instances_class_test!=1] = 0
print('test', np.mean(instances_class_test==y_pred))


mlflow.end_run()
# ?????????
#por razones de código. test_y y train_y tenían antes 
#valores de 0 a 1, y no 0 y 1?
"""
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
"""
"""
Log loss, aka logistic loss or cross-entropy loss.
This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, 
defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data 
y_true. The log loss is only defined for two or more labels

bag_results = bag_metrics(vgpmil_model, test_x, T_test, bag_index_test, 'VGPMIL')
print(bag_results)
"""
