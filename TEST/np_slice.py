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
list = [ [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16] ]
c = np.array(list)
res = c[[1,2], [1,0]]   # >> 0행 1열, 3행 3열
print(res)


print("=======================")
# 필요한 배열의 요소만 사용하기 위해
list = [ [1,2,3], [4,5,6], [7,8,9] ]
d = np.array(list)
b_arr = np.array([ [False, True, False], [True, False, True], [False, True, False] ])
res = d[b_arr]
print(res)