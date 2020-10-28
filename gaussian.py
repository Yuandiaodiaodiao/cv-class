import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def conv2_pixel(kernel, img, x, y):
    kh, kw = kernel.shape
    total = 0
    # print(x,y)
    for i in range(kh):
        for j in range(kw):
            total += img[i + x, y + j] * kernel[i][j]
    return total


def gen_gaussian_kernel(sigma):
    kernelSize = math.floor(6 * sigma - 1) // 2 * 2 + 1
    print(kernelSize)
    kernel = np.zeros((kernelSize, kernelSize))
    center = kernelSize // 2
    sumVal = 0
    for i in range(kernelSize):
        for j in range(kernelSize):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
            sumVal += kernel[i, j]
    kernel = kernel / sumVal
    print(kernel)
    return kernel


def conv2(kernel, img, center):
    h, w, channel = img.shape
    center1=kernel.shape[0]
    center2=kernel.shape[1]
    for c in range(channel):
        imgc = img[..., c]
        for i in range(center, h - center):
            for j in range(center, w - center):
                col = conv2_pixel(kernel, imgc, i - center, j - center)
                imgc[i, j] = round(col)
    return img

def gaussian_kernel1d(sigma):
    kernelSize = math.floor(6 * sigma - 1) // 2 * 2 + 1
    print(kernelSize)
    kernel = np.zeros(kernelSize)
    center = kernelSize // 2
    sumVal = 0
    for i in range(kernelSize):
            x= i - center
            kernel[i] = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (math.sqrt( 2 * math.pi) * sigma)
            sumVal += kernel[i]
    kernel = kernel / sumVal
    print(kernel)
    return kernel




def gaussianFast(img,sigma):
    h, w, _ = img.shape
    kernel = gaussian_kernel1d(sigma)
    kernel1=kernel.reshape((kernel.shape[0],1))
    kernel2=kernel.reshape((1,kernel.shape[0]))
    center = kernel.shape[0] // 2
    img = border_replicate(img, center)
    img = conv2(kernel1, img, center)
    img = conv2(kernel2, img, center)
    img = img[center:center + h, center:center + w, ...]
    return img

def gaussian(img, sigma):
    h, w, _ = img.shape
    kernel = gen_gaussian_kernel(sigma)
    center = kernel.shape[0] // 2
    img = border_replicate(img, center)
    img = conv2(kernel, img, center)
    img = img[center:center + h, center:center + w, ...]
    return img


def meanfilterChannel(img, size):
    S = np.zeros(img.shape, np.int)
    for i in range(S.shape[0]):
        for j in range(1, S.shape[1]):
            S[i, j] = S[i, j - 1] + np.sum(img[:i, j])
    w = (size - 1) // 2
    Z = (2 * w + 1) ** 2
    for i in range(w, S.shape[0] - w):
        for j in range(w, S.shape[1] - w):
            col = 1 / (Z) * (S[i + w, j + w] + S[i - w - 1, j - w - 1]
                             - S[i + w, j - w - 1] - S[i - w - 1, j + w])
            img[i, j] = col
    return img


def meanfilter(img, size):
    oh, ow, channel = img.shape
    img = border_replicate(img, size)
    for c in range(channel):
        img[..., c] = meanfilterChannel(img[..., c], size)

    img = img[size:size + oh, size:size + ow, ...]
    return img


def border_replicate(img, ranges):
    # 边缘拓展
    h, w, channel = img.shape
    newImage = np.zeros((h + ranges * 2, w + ranges * 2, channel), dtype=np.uint8)
    for channel in range(channel):
        for i in range(h):
            for j in range(w):
                newImage[i + ranges, j + ranges, channel] = img[i, j, channel]
        for i in range(ranges):
            for j in range(w + ranges * 2):
                newImage[i, j, channel] = newImage[ranges, j, channel]
        for i in range(h + ranges, h + ranges * 2):
            for j in range(w + ranges * 2):
                newImage[i, j, channel] = newImage[ranges + h - 1, j, channel]
        for i in range(h + ranges * 2):
            for j in range(ranges):
                newImage[i, j, channel] = newImage[i, ranges, channel]
        for i in range(h + ranges * 2):
            for j in range(w + ranges, w + ranges * 2):
                newImage[i, j, channel] = newImage[i, ranges + w - 1, channel]
    return newImage

def middlePixel(img,x,y,size):
    col=[]
    for i in range(size):
        for j in range(size):
            col.append(img[x+i,y+j])
    col.sort()
    return col[len(col)//2]
def middleFilter(img,size):
    h, w, channel = img.shape

    img = border_replicate(img, size)
    h2, w2, channel = img.shape
    center=size//2
    for c in range(channel):
        imgc = img[..., c]
        for i in range(center, h2 - center):
            for j in range(center, w2 - center):
                col = middlePixel(imgc,i-center,j-center,size)
                imgc[i, j] = col
    img = img[size:size + h, size:size + w, ...]
    return img

if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(231)
    plt.title("origin")
    plt.imshow(img)

    rows, cols, dims = img.shape

    for i in range(5000):
        x = np.random.randint(0, rows)

        y = np.random.randint(0, cols)

        img[x, y, :] = 255
    plt.subplot(232)
    plt.title("noise")
    plt.imshow(img)

    imgBorder = border_replicate(img, 50)
    plt.subplot(233)
    plt.title("border*50")
    plt.imshow(imgBorder)

    imgOut =gaussian(img, 1)
    plt.subplot(234)
    plt.title("gaussian5x5")
    plt.imshow(imgOut)

    imgmean = meanfilter(img, 5)
    plt.subplot(235)
    plt.title("meanfilter")
    plt.imshow(imgmean)


    imgmiddle=middleFilter(img,5)
    plt.subplot(236)
    plt.title("middle")
    plt.imshow(imgmiddle)
    plt.show()

# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# eq(img)
