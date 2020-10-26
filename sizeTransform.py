import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def resize(img,scalex,scaley):
    imgy,imgx,_=img.shape
    newx=math.ceil(scalex*imgx)
    newy=math.ceil(scaley*imgy)
    def getTargetPos(y,x):
        return math.floor(y/scaley),math.floor(x/scalex)
    newimage = np.zeros((newy,newx,img.shape[2]), dtype=np.uint8)
    def transform(img,x1,x2,y1,y2,x,y):
        # print((x1,x2,y1,y2,x,y))
        x=(x-x1)/(x2-x1)
        y=(y-y1)/(y2-y1)
        newcol=(img[y1][x1]*(1-x)*(1-y)
                +img[y1][x2]*x*(1-y)
                +img[y2][x1]*(1-x)*y
                +img[y2][x2]*x*y)
        return newcol
    for i in range(newy):
        for j in range(newx):
            y,x=getTargetPos(i,j)
            x1=math.floor(x)
            x2=min(math.ceil(x+0.001),imgx-1)
            y1=math.floor(y)
            y2=min(math.ceil(y+0.001),imgy-1)
            if x1==x2 : x1-=1
            if y1==y2 : y1-=1
            newimage[i][j]=transform(img,x1,x2,y1,y2,x,y)
    return newimage
if __name__=="__main__":

    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(img)
    newimg=resize(img,0.5,0.1)
    plt.subplot(132)
    plt.imshow(newimg)
    newimg = resize(img, 0.1, 0.5)
    plt.subplot(133)
    plt.imshow(newimg)
    plt.show()