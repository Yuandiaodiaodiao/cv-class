import numpy as np

import cv2

a = np.arange(256).reshape((32,8))

print("ori data: \n{}".format(a))

m,n = a.shape

print(m,n)

# Y = np.zeros(256).reshape((16,16))

hdata = np.vsplit(a,n/8) # 垂直分成高度度为8 的块

for i in range(0, n//8):
     blockdata = np.hsplit(hdata[i],m/8)
     #垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
     for j in range(0, m//8):
         block = blockdata[j]
         print("block[{},{}] data \n{}".format(i,j,blockdata[j]))
         Yb = cv2.dct(block.astype(np.float))
         # print("dct data\n{}".format(Yb))
         # iblock = cv2.idct(Yb)
         # print("idct data\n{}".format(iblock))