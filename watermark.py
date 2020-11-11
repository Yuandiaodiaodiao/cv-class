from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
#watermark
watermark_path="test.jpg"
watermark=cv2.imread(watermark_path)
watermark2=cv2.cvtColor(watermark,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
ret,waterMark =cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
h,w=waterMark.shape
#print(w,h)
#cv2.imshow("binary", dst)
#cv2.waitKey()

#plt.subplot(211)
#plt.imshow(img2)
plt.subplot(212)
plt.imshow(waterMark,'gray'),plt.title('watermark')

#plt.show()
#processing image
img=cv2.imread('test2.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(211)
plt.imshow(img),plt.title('origin')
plt.show()
#dct
img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#cut into 8*8 block
hi,wi=img_gray.shape
hi8=hi-hi%8
wi8=wi-wi%8
print(hi8,wi8)
img_gray=img_gray[0:hi8,0:wi8]
plt.imshow(img_gray,'gray')
plt.show()

#a,b=img_gray.shape
#print(a,b)

hdata=np.vsplit(img_gray,hi8/8)
for i in range(0, hi8//8):
     blockdata = np.hsplit(hdata[i],wi8/8)
     #8*8
     for j in range(0, wi8//8):
         block = blockdata[j]
         print("block[{},{}] data \n{}".format(i,j,blockdata[j]))
         Yb = cv2.dct(block.astype(np.float))
         print("dct data\n{}".format(Yb))
         iblock = cv2.idct(Yb)
         print("idct data\n{}".format(iblock))


