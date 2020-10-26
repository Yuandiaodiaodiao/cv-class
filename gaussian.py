import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def gaussian(img,sigma):
    kernelSize=math.floor(6*sigma-1)//2*2+1
    print(kernelSize)
    kernel = np.zeros((kernelSize, kernelSize))
    center=kernelSize//2
    sumVal=0
    for i in range(kernelSize):
        for j in range(kernelSize):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma**2))/(2*math.pi*sigma**2)
            sumVal += kernel[i, j]
    kernel=kernel/sumVal
    print(kernel)
    img

    return 0
if __name__=="__main__":
    img = cv2.imread("test.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.subplot(131)
    # plt.imshow(img)
    imgOut=gaussian(img,1)
    # plt.subplot(132)
    # plt.imshow(imgOut)

    # plt.show()
    #
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # eq(img)