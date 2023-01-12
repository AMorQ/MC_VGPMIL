import numpy as np

x_train = np.linspace(0, 1, 15)
third = int(len(x_train)/3)
y_train = np.array(third*[0] + third*[1] + third*[2])
bag_indexes  = np.array((third-2)*[0] + 1*[1] + 1*[2] + third*[1] +
third*[2])
bag_label = bag_indexes

for b in range(3):
   print(x_train[(bag_indexes==b)])
   print(y_train[bag_indexes==b])
   print(bag_indexes[bag_indexes==b])
