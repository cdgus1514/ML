import numpy as np

list = [ [1,2,3], [4,5,6], [7,8,9], [10,11,12]]
arr = np.array(list)

print(arr)
print("=======================")


a = arr[0:3, 0:1]   # [행 범위, 열 범위]
print(a)

print("=======================")
b = arr[1:, 2:]
print(b)

print("=======================")
list = [ [1,2,3,4], [5,6,6,7], [8,9,10,11], [12,13,14,15]]
c = np.array(list)
res = c[[0,2], [1,3]]
print(res)