import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


def reshape(img):
    imgy, imgx, _ = img.shape
    newimg = np.zeros(img.shape, dtype=np.uint8)
    newimg = np.copy(img)

    def normalize(value, max):
        half = 0.5 * max
        return (value - half) / (half)

    def renormalize(value, max):
        half = max * 0.5
        return round(value * half + half)

    for i in range(imgy):
        for j in range(imgx):
            nory = normalize(i, imgy)
            norx = normalize(j, imgx)
            r = math.sqrt(nory * nory + norx * norx)
            if r >= 1:
                newy, newx = nory, norx
            else:
                O = (1-r) * (1 - r)
                newx = math.cos(O) * norx - math.sin(O) * nory
                newy = math.sin(O) * norx + math.cos(O) * nory


            newy = renormalize(newy, imgy)
            newx = renormalize(newx, imgx)
            newimg[i][j] = img[newy][newx]
    return newimg


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    newimg = reshape(img)
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(newimg)
    plt.subplot(133)
    plt.imshow(newimg)
    plt.show()
