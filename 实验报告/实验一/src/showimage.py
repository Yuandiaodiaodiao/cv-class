import cv2
import matplotlib.pyplot as plt
img=cv2.imread("test.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
img2=cv2.imread("img.jpg")
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

