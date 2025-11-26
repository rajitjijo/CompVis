import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cimg = plt.imread("images/tulips.jpg")

print(cimg.shape)
print(cimg.dtype)

r = cimg[:,:,0]
g = cimg[:,:,1]
b = cimg[:,:,2]


# plt.imshow(cimg)
# plt.show()

plt.figure(1)
plt.subplot(2,3,1)
plt.imshow(cimg)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(cimg)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(cimg)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.imshow(r,cmap = "gray")
plt.xticks([])
plt.yticks([])
plt.title("RED")

plt.subplot(2,3,5)
plt.imshow(g, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title("GREEN")

plt.subplot(2,3,6)
plt.imshow(b, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title("BLUE")

plt.show()

#OPENCV reads in bgr instead of rgb

# cvimg = cv.imread("images/tulips.jpg")

