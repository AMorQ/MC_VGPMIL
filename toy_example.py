import numpy as np

def toy_example(N):
   x_train = np.linspace(0, 1, N)#total amount of items must be multiple of 3, but 3
   third = int(len(x_train)/3)
   y_train = np.array(third*[0] + third*[1] + third*[2])
   bag_indexes = np.array((third-2)*[0] + 1*[1] + 1*[2] + third*[1] +
   third*[2])
   bag_label = bag_indexes

   #for b in range(3):
   #   print(x_train[(bag_indexes==b)])
   #   print(y_train[bag_indexes==b])
   #   print(bag_indexes[bag_indexes==b])


   x_train = x_train.reshape(-1, 1)
   y_train = y_train.reshape(-1, 1)

   return x_train, y_train, bag_label

def very_toy_example():
   #we for a database of just two bags
   x_train = np.asarray([0, 0.33, 0.66, 1])
   y_train = np.asarray([0, 0, 0, 1])
   bag_indexes = [0, 0, 1, 1]
   bag_label = np.asarray(bag_indexes)
   x_train = x_train.reshape(-1, 1)
   y_train = y_train.reshape(-1, 1)
   return x_train, y_train, bag_label
