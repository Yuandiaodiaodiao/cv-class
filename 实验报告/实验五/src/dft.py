import numpy as np
import math


# 一维dft

def dft(array):
    return dftCore(1, array)


def idft(array):
    return dftCore(-1, array)


def dftCore(direction, array, deepCopy=False):
    output = np.copy(array).astype(np.complex)
    arrayLen = len(array)
    for i, outputItem in enumerate(np.nditer(output, op_flags=['readwrite'])):
        outputItem[...] = complex(0, 0)
        arg = -direction * 2.0 * math.pi * i / arrayLen
        for index, item in enumerate(np.nditer(array)):
            cosarg = math.cos(index * arg)
            sinarg = math.sin(index * arg)
            real = outputItem.real + item.real * cosarg - item.imag * sinarg
            imag = outputItem.imag + item.real * sinarg + item.imag * cosarg
            outputItem[...] = complex(real, imag)

    if direction == -1:
        for item in np.nditer(output, op_flags=['readwrite']):
            item[...] = complex(item.real / arrayLen, 0)
    if deepCopy:
        for originItem, outputItem in zip(np.nditer(array, op_flags=['readwrite']), np.nditer(output)):
            originItem[...] = outputItem
    return output


def dft2d(image):
    return dft2dCore(1, image)


def idft2d(image):
    return dft2dCore(-1, image)


def dft2dCore(direction, image):
    h, w = image.shape
    output = np.copy(image).astype(np.complex)
    for i in range(h):
        # 按行dft
        dftCore(direction, output[i, ...], True)
    for j in range(w):
        # 按列dft
        dftCore(direction, output[..., j], True)
    return output


def fft2d(image):
    return fft2dCore(1, image)


def ifft2d(image):
    return fft2dCore(-1, image)

#fft行列分离 各做一遍
def fft2dCore(direction, image):
    h, w = image.shape
    output = np.copy(image).astype(np.complex)
    for i in range(h):
        # 按行fft
        fftCore(direction, output[i, ...])
    for j in range(w):
        # 按列fft
        fftCore(direction, output[..., j])
    return output


def fftCore(direction, array):
    # 转换成实数
    if array.dtype!=np.complex:
        # 已经是complex就不用再astype了 否则引用会断
        array = array.astype(np.complex)
    m = math.floor(math.log2(len(array)))
    nn = 2 ** m
    i2 = nn >> 1
    j = 0
    # 二分 交换
    for i in range(nn - 1):
        if i < j:
            array[i], array[j] = array[j], array[i]
        k = i2
        while k <= j:
            j -= k
            k >>= 1
        j += k

    c1 = -1
    c2 = 0
    l2 = 1
    for l in range(m):
        l1 = l2
        l2 <<= 1
        u1 = 1
        u2 = 0
        for j in range(l1):
            for i in range(j, nn, l2):
                i1 = i + l1
                t1 = u1 * array[i1].real - u2 * array[i1].imag
                t2 = u1 * array[i1].imag + u2 * array[i1].real
                array[i1] = array[i] - complex(t1, t2)
                array[i] += complex(t1, t2)
            z = u1 * c1 - u2 * c2
            u2 = u1 * c2 + u2 * c1
            u1 = z
        c2 = math.sqrt((1 - c1) / 2)
        c1 = math.sqrt((1 + c1) / 2)
        # 只要确保fft和ifft的时候 有一个是c2=-c2就行
        if direction == 1:
            c2 = -c2
    # ifft
    if direction == -1:
        array /= nn

    return array


# fft准确性测试
def Testdft():
    arr = np.array([0, 1, 2, 3])
    ans1 = np.fft.fft(arr)
    ans2 = fftCore(1, arr)
    print("dft fft")
    print(ans1)
    print(ans2)
    ans3 = np.fft.ifft(ans1)
    ans4 = fftCore(-1, ans2)
    print("idft ifft")
    print(ans3)
    print(ans4)
    arr = np.array([[1, 2], [3, 4]])
    ans1 = np.fft.fft2(arr)
    ans2 = fft2d(arr)
    print("dft2 fft2")
    print(ans1)
    print(ans2)
    ans5=np.fft.fftshift(ans1)
    ans6=fftshift(ans2)
    print('np.shift shift')
    print(ans5)
    print(ans6)
    ans7=np.fft.ifftshift(ans5)
    ans8=fftshift(ans6)
    print('np.ishift ishift')
    print(ans7)
    print(ans8)
    ans3 = np.fft.ifft2(ans1)
    ans4 = ifft2d(ans2)
    print("idft2 ifft2")
    print(ans3)
    print(ans4)


# 归一化到0~1
def normalize(image):
    max = np.max(image)
    min = np.min(image)
    delta = max - min
    image = np.subtract(image, min)
    image = np.divide(image, delta)
    return image


# 0~1的图片缩放到0~255
def to255(img):
    img = np.multiply(img, 255)
    for item in np.nditer(img, op_flags=['readwrite']):
        if item < 0:
            item[...] = 0
        elif item > 255:
            item[...] = 255
    return img.astype(np.uint8)


def idealpass(img, D, arg):
    h, w = img.shape
    img = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            d = math.sqrt((i - h // 2) ** 2 + (j - w // 2) ** 2)
            if d > D:
                img[i, j] = arg
            else:
                img[i, j] = 1 - arg
    return img


def ButterworthPass(img, D, n, arg):
    h, w = img.shape
    img = np.zeros(img.shape)
    for i in range(h):
        for j in range(w):
            d = math.sqrt((i - h // 2) ** 2 + (j - w // 2) ** 2)
            hx = 1 / (1 + (d / D) ** (2 * n))
            img[i, j] = hx if arg == 0 else 1 - hx

    return img


# 对灰度图使用滤波器
def appendFilter(img, filterFunc):
    imgdft = fft2d(img)
    imgdft = np.fft.fftshift(imgdft)
    filter = filterFunc(img)
    imgdft *= filter
    imgdft = np.fft.ifftshift(imgdft)
    imgdft = ifft2d(imgdft)
    return imgdft



def fftshift(image):
    image=np.copy(image)
    h,w=image.shape
    for i in range(h//2):
        for j in range(w//2):
            # image[i,j]=complex(0)
            image[i,j],image[h//2+i,w//2+j]=image[h//2+i,w//2+j],image[i,j]
    for i in range(h//2,h):
        for j in range(w//2):
            image[i,j],image[i-h//2,w//2+j]=image[i-h//2,w//2+j],image[i,j]
    return image
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Testdft()
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orih, oriw = img.shape


    def reresize(img):
        return cv2.resize(img, (oriw, orih), cv2.INTER_CUBIC)


    # 适应fft的2**N大小
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    plt.subplot(331)
    plt.imshow(reresize(img), cmap='gray')

    # imgdft=dft2d(img)
    imgdft = fft2d(img)

    imgdft = fftshift(imgdft)
    # imgdft=ifftshift(imgdft)
    imgabs = np.log(np.abs(imgdft))

    imgangle = np.angle(imgdft)
    np.fft.ifftshift=fftshift
    plt.subplot(332)
    plt.title('amplitude')
    plt.imshow(reresize(imgabs), cmap='gray')
    plt.subplot(333)
    plt.title('phase')
    plt.imshow(reresize(imgangle), cmap='gray')

    plt.subplot(336)
    plt.imshow(reresize(np.real(ifft2d(fftshift(np.copy(imgdft))))),cmap='gray')

    lowpassimg = imgdft * idealpass(imgdft, 20, 0)
    lowpassimg = np.real(ifft2d(np.fft.ifftshift(lowpassimg)))
    plt.subplot(334)
    plt.title('lowpass')
    plt.imshow(reresize(lowpassimg), cmap='gray')

    highpassimg = imgdft * idealpass(imgdft, 20, 1)
    highpassimg = np.real(ifft2d(np.fft.ifftshift(highpassimg)))
    plt.subplot(335)
    plt.title("highpass")
    plt.imshow(reresize(highpassimg), cmap='gray')

    lowpassimg = imgdft * ButterworthPass(imgdft, 20, 2, 0)
    lowpassimg = np.real(ifft2d(np.fft.ifftshift(lowpassimg)))
    plt.subplot(337)
    plt.title("lowpass")
    plt.imshow(reresize(lowpassimg), cmap='gray')

    highpassimg = imgdft * ButterworthPass(imgdft, 20, 2, 1)
    highpassimg = np.real(ifft2d(np.fft.ifftshift(highpassimg)))
    plt.subplot(338)
    plt.title("highpass")
    plt.imshow(reresize(highpassimg), cmap='gray')
    plt.show()
