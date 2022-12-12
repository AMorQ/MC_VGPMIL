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
from helperfunctions import RBF
from mc_vgpmil import mc_vgpmil
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
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()#np.shape(train_x)= (60000, 28, 28)

    # Reshape and normalize data
    train_x = train_x.reshape(-1, 784).astype('float64')#np.shape(train_x): (60000, 784)
    test_x = test_x.reshape(-1, 784).astype('float64')
    train_y = train_y.reshape(-1,1)
    test_y = test_y.reshape(-1,1)
    
    train_x = train_x/255 #lo negro lo pone a rgb =(1,1,1) y lo blanco a rgb = (0,0,0)
    test_x = test_x/255

    scaler = StandardScaler()
    scaler.fit(train_x)
    #fit = Compute the mean and std to be used for later scaling.
    train_x = scaler.transform(train_x)
    #tranform = Perform standardization by centering and scaling
    test_x = scaler.transform(test_x)

    pca = PCA(n_components=30)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)

    
    mask_zeros=np.all([(train_y.ravel()!=1),(train_y.ravel()!=2),(train_y.ravel()!=3)],axis=0)
    mask_zeros_test = np.all([(test_y.ravel()!=1),(test_y.ravel()!=2),(test_y.ravel()!=3)],axis=0)
    zeros_train_x = train_x[mask_zeros]
    zeros_test_x = test_x[mask_zeros_test]
    np.random.shuffle(zeros_train_x)
    np.random.shuffle(zeros_test_x)

    threes_train_x =train_x[train_y.ravel()==3]
    threes_test_x = test_x[test_y.ravel()==3]
    np.random.shuffle(threes_train_x)
    np.random.shuffle(threes_test_x)
    
    twos_train_x = train_x[train_y.ravel()==2]
    twos_test_x = test_x[test_y.ravel()==2]
    np.random.shuffle(twos_train_x)
    np.random.shuffle(twos_test_x)
        
    ones_train_x = train_x[train_y.ravel()==1]
    ones_test_x = test_x[test_y.ravel() == 1]
    np.random.shuffle(ones_train_x)
    np.random.shuffle(ones_test_x)
    

      
    #CREATING BAGS
    """
    if we have Nzeros_train and 4 classes, class 0 will have bags with 9 zeros. The other
    three clases will have bags with 8 zeros. We want all the classes with the same number of bags
    C0_tr1+3*(8*Nbgas_class_train)=Nzeros_train
    NBags_class_train = Nzeros_train/(9+3*8)
    
    """
    nclasses=4
    Nzeros_train=len(zeros_train_x)
    Nzeros_test = len(zeros_test_x)
    
    NBags_class_train = math.floor(Nzeros_train/(9+(nclasses-1)*8))
    NBags_class_test = math.floor(Nzeros_test/(9+(nclasses-1)*8))
    
    C0_tr1 = NBags_class_train*9
    C0_ts1 = NBags_class_test*9
    C0_tr2 = C0_tr1+NBags_class_train*8
    C0_ts2 = C0_ts1+NBags_class_test*8
    C0_tr3 = C0_tr2+NBags_class_train*8
    C0_ts3 = C0_ts2+NBags_class_test*8
    C0_tr4 = C0_tr3 + NBags_class_train*8
    C0_ts4 = C0_ts3 + NBags_class_test*8
    
    
    zeros_train_x_C0 = zeros_train_x[0:C0_tr1]
    zeros_test_x_C0 = zeros_test_x[0:C0_ts1]
   
    zeros_train_x_C1 = zeros_train_x[C0_tr1:C0_tr2]
    zeros_test_x_C1 = zeros_test_x[C0_ts1:C0_ts2]
   
    zeros_train_x_C2 = zeros_train_x[C0_tr2:C0_tr3]
    zeros_test_x_C2 = zeros_test_x[C0_ts2:C0_ts3]
    
        
    zeros_train_x_C3 = zeros_train_x[C0_tr3:C0_tr4]
    zeros_test_x_C3 = zeros_test_x[C0_ts3:C0_ts4]
    
    zeros_train_x=zeros_train_x[0:C0_tr4]
    zeros_test_x = zeros_test_x[0:C0_ts4]
   
    bags_train= []
    bags_test = []
    multi_bags_train = []
    multi_bags_test = []
    bags_binary_class_train = []
    bags_binary_class_test = []
    
    #BAGS CLASS 0
    #creation of bags class 0 train
    j_tr = 0 #bag index 
    i_tr = 0 #instance index
    while i_tr+9<=int(len(zeros_train_x_C0)):# Counting how many C0 bags in train, and providing their id
        bags_train = np.hstack((bags_train, np.array(9*[j_tr]))) # Add the indexes
        i_tr = i_tr+9 
        j_tr  = j_tr+1 # Update the bag index
    zeros_train_x_C0=zeros_train_x_C0[0:i_tr]# I correct the dimension of zeros_trian_x_C0 arriving to the highest mulitple of 9   
    multi_bags_train = np.hstack((multi_bags_train, np.zeros(len(zeros_train_x_C0))))
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.zeros(len(zeros_train_x_C0))))
    #creation of bags class 0 test
    j_ts = 0 #bag index 
    i_ts = 0 #instance index
    while i_ts+9<=int(len(zeros_test_x_C0)):
        bags_test = np.hstack((bags_test, np.array(9*[j_ts]))) # Add the indexes
        i_ts = i_ts+9 
        j_ts  = j_ts+1 # Update the bag index
    zeros_test_x_C0=zeros_test_x_C0[0:i_ts]    
    multi_bags_test = np.hstack((multi_bags_test, np.zeros(len(zeros_test_x_C0))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.zeros(len(zeros_test_x_C0))))
        
    
    #BAGS CLASS 1 
    #creation of bags class 1 train (introducing zeros)
    i_tr=0
    while i_tr+8 <=int(len(zeros_train_x_C1)):
        bags_train = np.hstack((bags_train, np.array(8*[j_tr]))) # Add the indexes
        i_tr=i_tr+8
        j_tr = j_tr+1
    zeros_train_x_C1=zeros_train_x_C1[0:i_tr] 
    multi_bags_train = np.hstack((multi_bags_train, np.ones(len(zeros_train_x_C1))))
    bags_binary_class_train = np.hstack((bags_binary_class_train,np.ones(len(zeros_train_x_C1))))
    #creation of bags class 1 test (introducing zeros)
    i_ts=0
    while i_ts+8 <=int(len(zeros_test_x_C1)):
        bags_test = np.hstack((bags_test, np.array(8*[j_ts]))) # Add the indexes
        i_ts=i_ts+8
        j_ts  = j_ts+1
    zeros_test_x_C1=zeros_test_x_C1[0:i_ts] 
    multi_bags_test = np.hstack((multi_bags_test, np.ones(len(zeros_test_x_C1))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(zeros_test_x_C1))))
    #introducing class 1 instances in train bags 
    i_tr=0
    bags_ids= np.unique(bags_train[C0_tr1:C0_tr2])
    ones_train_x=ones_train_x[0:len(bags_ids)]
    while i_tr <int(len(ones_train_x)):
        bags_train = np.hstack((bags_train, np.array(1*bags_ids[i_tr])))
        i_tr=i_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 1*np.ones(len(ones_train_x))))
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.ones(len(ones_train_x))))
    #introducing class 1 instances in test bags 
    i_ts=0
    bags_ids= np.unique(bags_test[C0_ts1:C0_ts2])
    ones_test_x=ones_test_x[0:len(bags_ids)]
    while i_ts <int(len(ones_test_x)):
        bags_test = np.hstack((bags_test, np.array(1*bags_ids[i_ts])))
        i_ts=i_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 1*np.ones(len(ones_test_x))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(ones_test_x))))
          
    
    
    #BAGS CLASS 2   
    #creation of bags class 2 train
    i_tr=0
    while i_tr+8 <=int(len(zeros_train_x_C2)):
        bags_train = np.hstack((bags_train, np.array(8*[j_tr]))) # Add the indexes
        i_tr =i_tr+8
        j_tr = j_tr +1
    zeros_train_x_C2=zeros_train_x_C2[0:i_tr] 
    multi_bags_train = np.hstack((multi_bags_train, 2*np.ones(len(zeros_train_x_C2))))
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.ones(len(zeros_train_x_C2))))
    #creation of bags class 2 test
    i_ts=0
    while i_ts+8 <=int(len(zeros_test_x_C2)):
        bags_test = np.hstack((bags_test, np.array(8*[j_ts]))) # Add the indexes
        i_ts=i_ts+8
        j_ts = j_ts +1
    zeros_test_x_C2=zeros_test_x_C2[0:i_ts] 
    multi_bags_test = np.hstack((multi_bags_test, 2*np.ones(len(zeros_test_x_C2))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(zeros_test_x_C2))))
    #introducing class 2 train instances in bags 
    i_tr=0
    bags_ids= np.unique(bags_train[C0_tr2:C0_tr3])
    twos_train_x=twos_train_x[0:len(bags_ids)]
    while i_tr <int(len(twos_train_x)):
        bags_train = np.hstack((bags_train, np.array(1*[bags_ids[i_tr]])))
        i_tr=i_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 2*np.ones(len(twos_train_x))))
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.ones(len(twos_train_x))))
    #introducing class 2 instances in bags 
    i_ts=0
    bags_ids= np.unique(bags_test[C0_ts2:C0_ts3])
    twos_test_x=twos_test_x[0:len(bags_ids)]
    while i_ts <int(len(twos_test_x)):
        bags_test = np.hstack((bags_test, np.array(1*[bags_ids[i_ts]])))
        i_ts=i_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 2*np.ones(len(twos_test_x))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(twos_test_x))))
    
          
    
    
    #BAGS CLASS 3    
    #creation of bags class 3 train
    i_tr=0
    while i_tr+8 <=int(len(zeros_train_x_C3)):
        bags_train = np.hstack((bags_train, np.array(8*[j_tr]))) # Add the indexes
        i_tr=i_tr+8
        j_tr = j_tr + 1
    zeros_train_x_C3=zeros_train_x_C3[0:i_tr]
    multi_bags_train = np.hstack((multi_bags_train, 3*np.ones(len(zeros_train_x_C3))))
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.ones(len(zeros_train_x_C3))))
    #creation of bags class 3 test
    i_ts=0
    while i_ts+8 <=int(len(zeros_test_x_C3)):
        bags_test = np.hstack((bags_test, np.array(8*[j_ts]))) # Add the indexes
        i_ts=i_ts+8
        j_ts = j_ts+1
    zeros_test_x_C3=zeros_test_x_C3[0:i_ts]
    multi_bags_test = np.hstack((multi_bags_test, 3*np.ones(len(zeros_test_x_C3))))
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(zeros_test_x_C3))))
    #introduce class 3 train instances in bags 
    i_tr=0
    bags_ids= np.unique(bags_train[C0_tr3:C0_tr4])
    threes_train_x=threes_train_x[0:len(bags_ids)]
    while i_tr <int(len(threes_train_x)):
        bags_train = np.hstack((bags_train, np.array(1*[bags_ids[i_tr]])))
        i_tr=i_tr+1
    multi_bags_train = np.hstack((multi_bags_train, 3*np.ones(len(threes_train_x))))  
    bags_binary_class_train = np.hstack((bags_binary_class_train, np.ones(len(threes_train_x))))
    #introduce class 3 test instances in bags 
    i_ts=0
    bags_ids= np.unique(bags_test[C0_ts3:C0_ts4])
    threes_test_x=threes_test_x[0:len(bags_ids)]
    while i_ts <int(len(threes_test_x)):
        bags_test = np.hstack((bags_test, np.array(1*[bags_ids[i_ts]])))
        i_ts=i_ts+1
    multi_bags_test = np.hstack((multi_bags_test, 3*np.ones(len(threes_test_x))))  
    bags_binary_class_test = np.hstack((bags_binary_class_test, np.ones(len(threes_test_x))))
    
       
    
     
    
    train_x_matrix = np.concatenate((zeros_train_x_C0,zeros_train_x_C1,zeros_train_x_C2,zeros_train_x_C3,ones_train_x,twos_train_x,threes_train_x), axis = 0)
    test_x_matrix = np.concatenate((zeros_test_x_C0,zeros_test_x_C1,zeros_test_x_C2,zeros_test_x_C3,ones_test_x,twos_test_x,threes_test_x), axis = 0)
    #instaces_class vector con la clase de cada instancia
    instances_class_train = np.concatenate((np.zeros(len(zeros_train_x_C0)),\
            np.zeros(len(zeros_train_x_C1)),np.zeros(len(zeros_train_x_C2)),np.zeros(len(zeros_train_x_C3)),\
            np.ones(len(ones_train_x)),2*np.ones(len(twos_train_x)),3*np.ones(len(threes_train_x))))

    instances_class_test = np.concatenate((np.zeros(len(zeros_test_x_C0)),\
            np.zeros(len(zeros_test_x_C1)),np.zeros(len(zeros_test_x_C2)),np.zeros(len(zeros_test_x_C3)),\
            np.ones(len(ones_test_x)),2*np.ones(len(twos_test_x)),3*np.ones(len(threes_test_x))))

        
    bags_id_train = bags_train #vector with the bag id for every instance in train_x_matrix
    bags_id_test = bags_test #vector with the bag id for every instance in test_x_matrix
    bags_class_train= multi_bags_train # Cb - MC bag label- for the corresponding bag id of every instance in train_x_matrix
    bags_class_test= multi_bags_test #Cb - MC bag label- for the corresponding bag id of every instance in train_x_matrix
    
    return train_x_matrix, test_x_matrix, bags_id_train, bags_id_test, bags_binary_class_train,\
        bags_binary_class_test,bags_class_train,bags_class_test, instances_class_train, instances_class_test
    

def find_bag_elements(bag_id,array_bags):
    """
    To check a certain bag with bag_id = 3000 e.g.:
        pos=find_bag_elements(3000,bags_id)
        bags_class[pos1]
        instances_class[pos1]
        
    To check how many bags of each class we have
        counts, bins = np.histogram(multi_bags_train)
        plt.hist(bins[:-1], bins, weights=counts)
        

    """
    positions = []
    for i in range(len(array_bags)):
        print(i)
        print(int(array_bags[i]))
        if int(array_bags[i]) == bag_id:
           print('paso por aqui')
           input('Press enter to continue')
           print(i)
           print(int(i))
           positions = np.hstack((positions,i))
           positions = positions.astype(int)
    return positions