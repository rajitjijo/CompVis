import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# im = np.arange(10)
# print(im.shape)
# img = im[np.newaxis,:]
# print(img.shape)
# img_ = np.repeat(img, 100, axis=0)
# print(img_.shape)
# print(img_)
# plt.imshow(img_, cmap='gray')

#Processing Gray Scale Images

# img = plt.imread("images/albert-einstein_gray.jpg")
# print(img.shape)
# print(img.dtype)

# plt.imshow(img, cmap='gray')
# plt.show()


img_cv = cv.imread("images/albert-einstein_gray.jpg")

cv.imshow("gray", img_cv)
cv.waitKey(0)
cv.destroyAllWindows()
