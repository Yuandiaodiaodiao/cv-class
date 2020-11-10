import numpy as np
import matplotlib.pyplot as plt
sudoku=np.arange(256).reshape(32,8).astype(np.uint8)
import cv2
sudoku=cv2.imread("test4.png")
sudoku=cv2.cvtColor(sudoku,cv2.COLOR_BGR2GRAY)
print(sudoku)


plt.subplot(241)
plt.imshow(sudoku,cmap='gray')
def splitImage(img):
    #切分块大小
    spliteSize=8
    h,w=img.shape
    shape = (h//spliteSize,w//spliteSize,spliteSize,spliteSize)
    strides=(spliteSize*spliteSize*(w//spliteSize),spliteSize,spliteSize*(w//spliteSize),1)
    squares = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    sh,sw,_,_=squares.shape
    for i in range(sh):
        for j in range(sw):
            eightxeight=squares[i,j]
            # eightxeight就是一个小块了
            # cv2.dct(eightxeight)




    plt.subplot(245)
    i1=squares[0,0,...]
    print(i1)
    plt.imshow(squares[0,0,...],cmap='gray')
    plt.subplot(246)
    i2=squares[0, 1, ...]
    print(i2)
    plt.imshow(i2, cmap='gray')
    plt.subplot(247)
    i3=squares[1, 0, ...]
    i3[0,0]=0
    print(i3)
    plt.imshow(i3, cmap='gray')
    plt.subplot(248)
    i4=squares[1, 1, ...]
    i4[0,0]=0
    print(i4)
    plt.imshow(squares[1, 1, ...], cmap='gray')
    plt.show()

splitImage(sudoku)
def func():
    return 1,2


x,y=func()
y,x=x,y
