import numpy as np

list = [ [1,2,3], [4,5,6], [7,8,9] ]
arr = np.array(list)

print(arr)
print("=======================")


a = arr[0:3, 1:3]   # 리스트 개수, 범위
print(a)
