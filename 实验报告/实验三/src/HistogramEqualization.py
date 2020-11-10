import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def timeShow(fn):
    def _wrapper(*args, **kwargs):
        start = time.time()
        ans = fn(*args, **kwargs)
        print("%s cost %s second" % (fn.__name__,time.time() - start))
        return ans

    return _wrapper


from collections import Counter

@timeShow
def eq(img):
    h, w = img.shape
    pxNum = [0]*256
    def add(x):
        pxNum[x]+=1
    f = np.vectorize(add)
    f(img)
    pxNum = np.array(pxNum, dtype=np.float)
    pxNum *= (256-1) / (h * w)
    pxSum = np.cumsum(pxNum)
    colorMap = np.around(pxSum).astype(np.uint8)
    colorMap=list(colorMap)
    return colorMap


@timeShow
def HisEq(image):
    _, _, channel = image.shape
    channelMap = [eq(image[..., i]) for i in range(channel)]
    for channelId, map in enumerate(channelMap):
        f = np.vectorize(lambda x: map[x])
        image[..., channelId] = f(image[..., channelId])
    return image


if __name__ == "__main__":
    img = cv2.imread("test.jpg")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgOut = HisEq(np.copy(img))

    @timeShow
    def cv2Eq(img):
        _, _, channel = img.shape
        for channelId in range(channel):
            img[...,channelId]=cv2.equalizeHist(img[...,channelId])
        return img
    imgOut2=cv2Eq(np.copy(img))



    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(imgOut)
    plt.subplot(133)
    plt.imshow(imgOut2)
    plt.show()
    #
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # eq(img)
