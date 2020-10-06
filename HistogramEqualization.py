import cv2
import numpy as np

def eq(img):
    h,w=img.shape
    print(w,h,img.shape)
    pxNum=np.zeros((256))
    print(pxNum)
    for item in np.nditer(img):
        pxNum[item]+=1
    print(pxNum)
if __name__=="__main__":
    img = cv2.imread("test.jpg")

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eq(img)