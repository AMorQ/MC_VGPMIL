# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:04:39 2022

@author: luzgm


"""
import keras
import gpflow
import numpy as np
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import f1_score, log_loss
from sklearn.utils import shuffle
from helperfunctions import RBF
from mcv import mc_vgpmil

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_bags_multiclass(database):
    """
    This fuction returns:
        train_x_matrix: train data set of pca components for all instances in database
        test_x_matrix: test data set of pca components for all instances in database
        bags_id_train: vector with the bag id for every instance in train_x_matrix
        bags_id_test: vector with the bag id for every instance in test_x_matrix
        bags_binary_class_train: Tb -binary bag label- for the corresponding bag id of every instance in\
            train_x_matrix
        bags_binary_class_test: Tb -binary bag label- for the corresponding bag id of every instance in\
            test_x_matrix
        bags_class_train: Cb - MC bag label- for the corresponding bag id of every instance in\
            train_x_matrix
        bags_class_test: Cb - MC bag label- for the corresponding bag id of every instance in\
            test_x_matrix
        instances_class_train: MC  instance label for every instance in train_x_matrix
        instances_class_test: MC  instance label for every instance in train_x_matrix
    """
    # Load data from keras
    (train_x_orig, train_y_orig), (test_x_orig, test_y_orig) = keras.datasets.mnist.load_data()#np.shape(train_x)= (60000, 28, 28)

    # Reshape and normalize data
    train_x = train_x_orig.reshape(-1, 784).astype('float64')#np.shape(train_x): (60000, 784)
    test_x = test_x_orig.reshape(-1, 784).astype('float64')
    train_y = train_y_orig.reshape(-1,1)
    test_y = test_y_orig.reshape(-1,1)
    
    train_x = train_x/255 
    test_x = test_x/255

    scaler = StandardScaler()
    scaler.fit(train_x)
    #fit = Compute the mean and std to be used for later scaling
    train_x = scaler.transform(train_x)
    #tranform = Perform standarization by centering and scaling
    test_x = scaler.transform(test_x)

    pca = PCA(n_components=30)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)

    
    #mask_zeros=np.all([(train_y.ravel()!=1),(train_y.ravel()!=2),(train_y.ravel()!=3)],axis=0)
    mask_zeros=np.all([(train_y.ravel()==0)],axis=0)
    #mask_zeros_test = np.all([(test_y.ravel()!=1),(test_y.ravel()!=2),(test_y.ravel()!=3)],axis=0)
    mask_zeros_test = np.all([(test_y.ravel()==0)],axis=0)
    
    zeros_train_x = train_x[mask_zeros]
    zeros_train_x_orig = train_x_orig[mask_zeros]
    
    zeros_test_x = test_x[mask_zeros_test]
    zeros_test_x_orig = test_x_orig[mask_zeros_test]
    
    zeros_train_x,zeros_train_x_orig = shuffle(zeros_train_x,zeros_train_x_orig)
    zeros_test_x,zeros_test_x_orig = shuffle(zeros_test_x,zeros_test_x_orig)

    threes_train_x =train_x[train_y.ravel()==3]
    threes_train_x_orig = train_x_orig[train_y.ravel()==3]
    
    threes_test_x = test_x[test_y.ravel()==3]
    threes_test_x_orig = test_x_orig[test_y.ravel()==3]
    
    threes_train_x, threes_train_x_orig = shuffle(threes_train_x, threes_train_x_orig)
    threes_test_x, threes_test_x_orig = shuffle(threes_test_x, threes_test_x_orig)
    
    twos_train_x = train_x[train_y.ravel()==2]
    twos_train_x_orig = train_x_orig[train_y.ravel()==2]
    
    twos_test_x = test_x[test_y.ravel()==2]
    twos_test_x_orig = test_x_orig[test_y.ravel()==2]
    
    twos_train_x, twos_train_x_orig = shuffle(twos_train_x,twos_train_x_orig)
    twos_test_x,twos_test_x_orig = shuffle(twos_test_x,twos_test_x_orig)

    ones_train_x = train_x[train_y.ravel()==1]
    ones_train_x_orig = train_x_orig[train_y.ravel()==1]
    
    ones_test_x = test_x[test_y.ravel() == 1]
    ones_test_x_orig = test_x_orig[test_y.ravel() == 1]
    ones_train_x,ones_train_x_orig = shuffle(ones_train_x,ones_train_x_orig)
    ones_test_x, ones_test_x_orig = shuffle(ones_test_x,ones_test_x_orig)

    #CREATING BAGS
    """
    if we have Nzeros_train and 4 classes, class 0 will have bags with 9 zeros. The other
    three clases will have bags with 5 zeros and 4 instaces of the class. We want all the classes with the same number of bags
    C0_tr1+3*(8*Nbgas_class_train)=Nzeros_train
    NBags_class_train = Nzeros_train/(9+3*8)

    """
    nclasses=4
    Nzeros_train=len(zeros_train_x)
    Nzeros_test = len(zeros_test_x)
    
    #we decide to have the same number of bags (Nbags_class_train, Nbags_class_test) for each class
    #the limitation of number of bags per class in the database is given by the total number of zeros existing
    NBags_class_train = math.floor(Nzeros_train/(9+(nclasses-1)*5))
    NBags_class_test = math.floor(Nzeros_test/(9+(nclasses-1)*5))
    
    
    
    C0_tr1 = NBags_class_train*9 #indexes of the instances zero in train class 1
    C0_ts1 = NBags_class_test*9 #indextes of the instances zero in test class 1
    C0_tr2 = C0_tr1+ NBags_class_train*5 #indexes of the instances zero in train class 2
    C0_ts2 = C0_ts1+NBags_class_test*5 #indextes of the instances zero in test class 2
    C0_tr3 = C0_tr2+NBags_class_train*5 #indexes of the instances zero in train class 3
    C0_ts3 = C0_ts2+NBags_class_test*5 #indexes of the instances zero in test class 3
    C0_tr4 = C0_tr3 + NBags_class_train*5  #indexes of the instances zero in train class 4
    C0_ts4 = C0_ts3 + NBags_class_test*5  #indexes of the instances zero in test class 4
    
    
    zeros_train_x_C0 = zeros_train_x[0:C0_tr1]
    zeros_train_x_C0_orig = zeros_train_x_orig[0:C0_tr1]
    
    zeros_test_x_C0 = zeros_test_x[0:C0_ts1]
    zeros_test_x_C0_orig = zeros_test_x_orig[0:C0_ts1]
   
    zeros_train_x_C1 = zeros_train_x[C0_tr1:C0_tr2]
    zeros_train_x_C1_orig = zeros_train_x_orig[C0_tr1:C0_tr2]
    
    zeros_test_x_C1 = zeros_test_x[C0_ts1:C0_ts2]
    zeros_test_x_C1_orig = zeros_test_x_orig[C0_ts1:C0_ts2]
   
    zeros_train_x_C2 = zeros_train_x[C0_tr2:C0_tr3]
    zeros_train_x_C2_orig = zeros_train_x_orig[C0_tr2:C0_tr3]
    
    zeros_test_x_C2 = zeros_test_x[C0_ts2:C0_ts3]
    zeros_test_x_C2_orig = zeros_test_x_orig[C0_ts2:C0_ts3]
    
        
    zeros_train_x_C3 = zeros_train_x[C0_tr3:C0_tr4]
    zeros_train_x_C3_orig = zeros_train_x_orig[C0_tr3:C0_tr4]
    
    zeros_test_x_C3 = zeros_test_x[C0_ts3:C0_ts4]
    zeros_test_x_C3_orig = zeros_test_x_orig[C0_ts3:C0_ts4]
    
    zeros_train_x=zeros_train_x[0:C0_tr4]
    zeros_train_x_orig=zeros_train_x_orig[0:C0_tr4]
    
    zeros_test_x = zeros_test_x[0:C0_ts4]
    zeros_test_x_orig = zeros_test_x_orig[0:C0_ts4]
   
    bags_train= [] #vector of dim number of instances in train with bag id for each instance train
    bags_test = [] #vector of dim number of instances in test with bag id for each instance test
    multi_bags_train = []#vector of dim number of instances in train with bag mc label for each instance
    multi_bags_test = []#vector of dim number of instances in test with bag mc label for each instance
   
    j_tr = 0 #bag index along all the classes 
    j_ts = 0 #bag index 
    
    
    #BAGS CLASS 0
    #creation of bags class 0 train
    i_tr = 0 #instance index
    while i_tr+9<=int(len(zeros_train_x_C0)):# Counting how many C0 bags in train, and providing their id
        bags_train = np.hstack((bags_train, np.array(9*[j_tr]))) # Add the indexes
        i_tr = i_tr+9 
        j_tr  = j_tr+1 # Update the bag index
    multi_bags_train = np.hstack((multi_bags_train, np.zeros(len(zeros_train_x_C0))))
    #creation of bags class 0 test
    i_ts = 0 #instance index
    while i_ts+9<=int(len(zeros_test_x_C0)):
        bags_test = np.hstack((bags_test, np.array(9*[j_ts]))) # Add the indexes
        i_ts = i_ts+9 
        j_ts  = j_ts+1 # Update the bag index
    multi_bags_test = np.hstack((multi_bags_test, np.zeros(len(zeros_test_x_C0))))
    print('clase 0')       
    print(j_tr)
    print(j_ts)
    print(len(bags_train))
    print(len(bags_test))
   
   
    #BAGS CLASS 1 
    #creation of bags class 1 train (introducing zeros)
    i_tr=0
    while i_tr+5 <=int(len(zeros_train_x_C1)):
        bags_train = np.hstack((bags_train, np.array(5*[j_tr]))) # Add the indexes
        i_tr=i_tr+5
        j_tr = j_tr+1
    multi_bags_train = np.hstack((multi_bags_train, np.ones(len(zeros_train_x_C1))))
    #creation of bags class 1 test (introducing zeros)
    i_ts=0
    while i_ts+5 <=int(len(zeros_test_x_C1)):
        bags_test = np.hstack((bags_test, np.array(5*[j_ts]))) # Add the indexes
        i_ts=i_ts+5
        j_ts  = j_ts+1
    multi_bags_test = np.hstack((multi_bags_test, np.ones(len(zeros_test_x_C1))))
    
    #introducing class 1 instances in train bags 
    i_tr=0
    b_tr=0
    #bags_ids= np.unique(bags_train[C0_tr1:C0_tr2])
    bags_ids= np.unique(bags_train[len(bags_train)-1230:len(bags_train)])
    ones_train_x=ones_train_x[0:len(bags_ids)*4]# reshaping ones_train to use only the number of 'ones needed' we introduce 4 ones on each bag
    ones_train_x_orig=ones_train_x_orig[0:len(bags_ids)*4]
    while i_tr+4 <= int(len(ones_train_x)):
        bags_train = np.hstack((bags_train, np.array(4*[bags_ids[b_tr]])))
        i_tr=i_tr+4
        b_tr=b_tr+1
    multi_bags_train = np.hstack((multi_bags_train, np.ones(len(ones_train_x))))
    #introducing class 1 instances in test bags 
    i_ts=0
    b_ts=0
    #bags_ids= np.unique(bags_test[C0_ts1:C0_ts2])
    bags_ids= np.unique(bags_test[len(bags_test)-200:len(bags_test)])
    ones_test_x=ones_test_x[0:len(bags_ids)*4]
    ones_test_x_orig=ones_test_x_orig[0:len(bags_ids)*4]
    while i_ts+4 <=int(len(ones_test_x)):
        bags_test = np.hstack((bags_test, np.array(4*[bags_ids[b_ts]])))
        i_ts=i_ts+4
        b_ts=b_ts+1
    multi_bags_test = np.hstack((multi_bags_test, np.ones(len(ones_test_x))))
    
    
    
    print('clase 1')       
    print(j_tr)
    print(j_ts)
    print(len(bags_train))
    print(len(bags_test))
   

        
    #BAGS CLASS 2   
    #creation of bags class 2 train. Introducing zeros
    i_tr=0
    while i_tr+5 <=int(len(zeros_train_x_C2)):
        bags_train = np.hstack((bags_train, np.array(5*[j_tr]))) # Add the indexes
        i_tr =i_tr+5
        j_tr = j_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 2*np.ones(len(zeros_train_x_C2))))
    #creation of bags class 2 test. Introducing zeros
    i_ts=0
    while i_ts+5 <=int(len(zeros_test_x_C2)):
        bags_test = np.hstack((bags_test, np.array(5*[j_ts]))) # Add the indexes
        i_ts=i_ts+5
        j_ts = j_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 2*np.ones(len(zeros_test_x_C2))))
   
   
    
    #introducing class 2 train instances in bags 
    i_tr=0
    b_tr=0
    #bags_ids= np.unique(bags_train[C0_tr2:C0_tr3])
    bags_ids= np.unique(bags_train[len(bags_train)-1230:len(bags_train)])
    twos_train_x=twos_train_x[0:len(bags_ids)*4]
    twos_train_x_orig=twos_train_x_orig[0:len(bags_ids)*4]
    while i_tr+4<=int(len(twos_train_x)):
        bags_train = np.hstack((bags_train, np.array(4*[bags_ids[b_tr]])))
        i_tr=i_tr+4
        b_tr=b_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 2*np.ones(len(twos_train_x))))
    #introducing class 2 test instances in bags 
    i_ts=0
    b_ts=0
    #bags_ids= np.unique(bags_test[C0_ts2:C0_ts3])
    bags_ids= np.unique(bags_test[len(bags_test)-200:len(bags_test)])
    twos_test_x=twos_test_x[0:len(bags_ids)*4]
    twos_test_x_orig=twos_test_x_orig[0:len(bags_ids)*4]
    while i_ts+4<=int(len(twos_test_x)):
        bags_test = np.hstack((bags_test, np.array(4*[bags_ids[b_ts]])))
        i_ts=i_ts+4
        b_ts=b_ts+1
        #print(b_ts)
        #print(bags_ids[b_ts])
    multi_bags_test = np.hstack((multi_bags_test, 2*np.ones(len(twos_test_x))))
         
  
    print('clase 2')       
    print(j_tr)
    print(j_ts)
    print(len(bags_train))
    print(len(bags_test))
   
    
    #BAGS CLASS 3    
    #creation of bags class 3 train
    i_tr=0
    while i_tr+5 <=int(len(zeros_train_x_C3)):
        bags_train = np.hstack((bags_train, np.array(5*[j_tr]))) # Add the indexes
        i_tr=i_tr+5
        j_tr = j_tr + 1
    multi_bags_train = np.hstack((multi_bags_train, 3*np.ones(len(zeros_train_x_C3))))
    #creation of bags class 3 test , introducing zeros
    i_ts=0
    while i_ts+4 <=int(len(zeros_test_x_C3)):
        bags_test = np.hstack((bags_test, np.array(4*[j_ts]))) # Add the indexes
        i_ts=i_ts+4
        j_ts = j_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 3*np.ones(len(zeros_test_x_C3))))
    
    #introduce class 3 train instances in bags 
    i_tr=0
    b_tr=0
    #bags_ids= np.unique(bags_train[C0_tr3:C0_tr4])
    bags_ids= np.unique(bags_train[len(bags_train)-1230:len(bags_train)])
    threes_train_x=threes_train_x[0:len(bags_ids)*4]
    threes_train_x_orig=threes_train_x_orig[0:len(bags_ids)*4]
    while i_tr+4 <=int(len(threes_train_x)):
        bags_train = np.hstack((bags_train, np.array(4*[bags_ids[b_tr]])))
        i_tr=i_tr+4
        b_tr=b_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 3*np.ones(len(threes_train_x))))  
    #introduce class 3 test instances in bags 
    i_ts=0
    b_ts=0
    #bags_ids= np.unique(bags_test[C0_ts3:C0_ts4])
    bags_ids= np.unique(bags_test[len(bags_test)-200:len(bags_test)])
    threes_test_x=threes_test_x[0:len(bags_ids)*4]
    threes_test_x_orig=threes_test_x_orig[0:len(bags_ids)*4]
    while i_ts+4 <=int(len(threes_test_x)):
        bags_test = np.hstack((bags_test, np.array(4*[bags_ids[b_ts]])))
        i_ts=i_ts+4
        b_ts=b_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 3*np.ones(len(threes_test_x))))  
    
    print('clase 3')       
    print(j_tr)
    print(j_ts)
    print(len(bags_train))
    print(len(bags_test))
    
     
    
    train_x_matrix = np.concatenate((zeros_train_x_C0,zeros_train_x_C1,ones_train_x,zeros_train_x_C2,twos_train_x,zeros_train_x_C3,threes_train_x), axis = 0)
    train_x_matrix_orig = np.concatenate((zeros_train_x_C0_orig,zeros_train_x_C1_orig,ones_train_x_orig,zeros_train_x_C2_orig,twos_train_x_orig,zeros_train_x_C3_orig,threes_train_x_orig), axis = 0)
   
    test_x_matrix = np.concatenate((zeros_test_x_C0,zeros_test_x_C1,ones_test_x,zeros_test_x_C2,twos_test_x,zeros_test_x_C3,threes_test_x), axis = 0)
    test_x_matrix_orig = np.concatenate((zeros_test_x_C0_orig,zeros_test_x_C1_orig,ones_test_x_orig,zeros_test_x_C2_orig,twos_test_x_orig,zeros_test_x_C3_orig,threes_test_x_orig), axis = 0)
    
        
    instances_class_train = np.concatenate((np.zeros(len(zeros_train_x_C0)),\
            np.zeros(len(zeros_train_x_C1)), np.ones(len(ones_train_x)), np.zeros(len(zeros_train_x_C2)),2*np.ones(len(twos_train_x)),np.zeros(len(zeros_train_x_C3)),\
            3*np.ones(len(threes_train_x))))    
        

    instances_class_test = np.concatenate((np.zeros(len(zeros_test_x_C0)),\
            np.zeros(len(zeros_test_x_C1)), np.ones(len(ones_test_x)),np.zeros(len(zeros_test_x_C2)),2*np.ones(len(twos_test_x)),np.zeros(len(zeros_test_x_C3)),\
           3*np.ones(len(threes_test_x))))

        
    bags_id_train = bags_train #vector with the bag id for every instance in train_x_matrix
    bags_id_test = bags_test #vector with the bag id for every instance in test_x_matrix
    bags_class_train= multi_bags_train # Cb - MC bag label- for the corresponding bag id of every instance in train_x_matrix
    bags_class_test= multi_bags_test #Cb - MC bag label- for the corresponding bag id of every instance in train_x_matrix
    
    #instances_class_train : vector with multi class labels for all intaces in train_x_matrix
    #instaces_class_test: vector with multi class labels for all intances in test_x_matrix 
    
    return train_x_matrix, train_x_matrix_orig, test_x_matrix, test_x_matrix_orig, bags_id_train, bags_id_test, \
        bags_class_train,bags_class_test, instances_class_train, instances_class_test
    
def plot_bag(bag_id,original_data,bags):
    """
    Function to check that bags are well defined
    Parameters
    ----------
    bag_id : id of the bag to plot
    original data : train_x_orig or test_x_orig ( Mnist images before transformation)
    bags : vector with the bag id for every instance in original data
    Returns
    -------
    None.

    """

    mask = np.where(bags == bag_id)
    print(np.shape(original_data[mask]))
          
    for i in range(len(original_data[mask])):
        print(i)
        image = original_data[mask[0][i]][:,:]
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.show()

    return
 
