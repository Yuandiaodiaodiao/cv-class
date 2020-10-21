import cv2
import numpy as np
import matplotlib.pyplot as plt

def eq(img):
    h,w=img.shape
    print(w,h,img.shape)
    pxNum=np.zeros((256),dtype=np.uint)
    for item in np.nditer(img):
        pxNum[item]+=1
    print(pxNum)
    colorMap=np.zeros((256),dtype=np.uint)
    pSum=0
    totalPixel=h*w
    for i in range(256):
        pSum+=(256-1)*pxNum[i]/totalPixel
        colorMap[i]=round(pSum)
    print(colorMap)
    return colorMap
def HisEq(image):
    h,w,channel=image.shape
    channelMap=[]
    for i in range(channel):
        colorMap=eq(image[:,:,i])
        channelMap.append(colorMap)
    for i in range(h):
        for j in range(w):
            for k in range(channel):
                color=image[i][j][k]
                image[i][j][k]=channelMap[k][color]

    return image
if __name__=="__main__":
    img = cv2.imread("3.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(img)
    imgOut=HisEq(img)
    plt.subplot(132)
    plt.imshow(imgOut)

    plt.show()
    #
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # eq(img)