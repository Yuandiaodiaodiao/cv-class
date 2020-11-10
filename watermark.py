from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

#############watermark###############
watermark_path = "watermark.jpg"
watermark = cv2.imread(watermark_path)
watermark2 = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
ret, waterMark = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # Binarization
h, w = waterMark.shape
# print(w,h)
# cv2.imshow("binary", dst)
# cv2.waitKey()

# plt.subplot(211)
# plt.imshow(img2)
plt.subplot(212)
plt.imshow(waterMark, 'gray'), plt.title('watermark')

# plt.show()
#######################processing image###########
img = cv2.imread('timg.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(211)
plt.imshow(img), plt.title('origin')
plt.show()
# dct
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
##########################watermark prepare############
I = waterMark
alpha = 30
k1 = np.random.randn(8)
k2 = np.random.randn(8)
#####################cut into 8*8 block
hi, wi = img_gray.shape
hi8 = hi - hi % 8
wi8 = wi - wi % 8
print(hi8, wi8)
img_gray = img_gray[0:hi8, 0:wi8]
plt.subplot(121)
plt.imshow(img_gray, 'gray')

m, n = img_gray.shape
print(m, n)

hdata = np.vsplit(img_gray, m / 8)
# 一个m//8的[]数组用来接结果
ans=[[] for i in range(0,m//8)]
for i in range(0, m // 8):
    lineData = hdata[i]
    blockdata = np.hsplit(lineData, n // 8)
    for j in range(0, n // 8):
        block = blockdata[j]
        Yb = cv2.dct(block.astype(np.float))
        #存结果
        ans[i].append(Yb)

#每行把自己拼成一个8*n的
ans=map(lambda x:np.hstack(tuple(x)),ans)
#8*n之间做合并 得到m*n
ans=np.vstack(tuple(ans))


print(ans.shape)
plt.subplot(122)
plt.imshow(ans, 'gray')
plt.show()
###############add watermark
# for p in range(0,w):
#    for q in range(0,h):
#        x=p*8
#        y=q*8
