import numpy as np

my_list1 = [10, 'hello, world', 20]
print(my_list1)

my_arr1 = np.array(my_list1)
print(my_arr1)

my_list2 = [[12, 23, 33], [33, 44, 55], [66, 77, 88]]
print(my_list2)

my_arr2 = np.array(my_list2)
print(my_arr2)
print(np.sum(my_arr2))