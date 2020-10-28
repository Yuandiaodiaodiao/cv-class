import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
img=cv2.imread("test.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
def doGamma(img,gamma):
    gamma_img1 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                gamma_img1[i, j, k] = int(math.floor(min(1,math.pow(img[i, j, k]/255, gamma))*255))
    return gamma_img1
plt.subplot(231)
gamma1=doGamma(img,0.5)
plt.imshow(gamma1)
plt.subplot(232)
gamma2=doGamma(img,1)
plt.imshow(gamma2)
plt.subplot(233)
gamma3=doGamma(img,2)
plt.imshow(gamma3)
# plt.show()

# img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# cv2.imshow("0",img)
# cv2.waitKey()
shape=img.shape
print(shape)
# img=img/255
def dolog(img,logArg):
    c=logArg
    logimage = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                col=c*(math.log(1.0+img[i,j,k]/255.0,2))
            # print(c*math.log(255+255,500))
            # if col>1:col=1
                logimage[i][j][k] = math.floor(min(1    ,col)*255)
    return logimage
# img=np.where(255<=img,255,img)
logimg1=dolog(img,0.5)
plt.subplot(234)
plt.imshow(logimg1)
logimg2=dolog(img,1)
plt.subplot(235)
plt.imshow(logimg2)
logimg3=dolog(img,2)
plt.subplot(236)
plt.imshow(logimg3)
plt.show()

