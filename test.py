import numpy as np
import matplotlib.pyplot as plt
sudoku=np.arange(256).reshape(16,16).astype(np.uint8)
print(sudoku)


plt.subplot(241)
plt.imshow(sudoku,cmap='gray')
def splitImage(img):

    h,w=img.shape
    shape = (h//2,w//2,8,8)
    strides=(8*8*(w//8),8,8*(w//8),1)
    squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
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
    plt.imshow(squares[1, 1, ...], cmap='gray')
    plt.show()

splitImage(sudoku)