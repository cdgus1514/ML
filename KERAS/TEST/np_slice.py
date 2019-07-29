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










print("\n\n\n######################################## 차원변환 ########################################")
# list = [ [1,2,3], [4,5,6], [7,8,9], [10,11,12]]
test = np.linspace(1, 100, num=100)

print("\n\n###### 차원 변경 전 ######\n", test.shape)
print("###### test ######\n", test)                        # (100, )


test2 = test.reshape(4 ,25)
print("\n\n###### 2차원 변경 후 ######\n", test2.shape)
print("###### test2 ######\n", test2)                      # (4,25)


test3 = test2.reshape(4, 5, 5)
print("\n\n###### 3차원 변경 후 ######\n", test3.shape)
print("###### test3 ######\n", test3)                      # (4,5,5)