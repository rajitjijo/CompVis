import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2 as cv

def displayImageInActualSize(I):
    dpi = mpl.rcParams['figure.dpi']
    H,W = I.shape
    figSize = W/float(dpi) , H/float(dpi)
    fig = plt.figure(figsize = figSize)
    ax = fig.add_axes([0,0,1,1])
    ax.axis('off')
    ax.imshow(I,cmap='gray')
    plt.show()

grayimg = "../introduction/images/albert-einstein_gray.jpg"
colorimg = "../introduction/images/tulips.jpg"

gray = cv.imread(grayimg, cv.IMREAD_GRAYSCALE)
color = cv.cvtColor(cv.imread(colorimg), cv.COLOR_BGR2RGB)

# plt.imshow(color)
# plt.show()

# gray_resized = cv.resize(gray, fx=2, fy=0.5, dsize=None)

# p = np.array4,1])
# p_dash = scaling_mat.dot(p)

# print(p_dash)

numrows = gray.shape[0]
numcols = gray.shape[1]

sx, sy = 2,2
scaling_mat = np.array([[sx,0],[0,sy]])
gray3 = np.zeros((numrows*2, numcols*2), dtype='uint8')

for i in range(numrows):
    for j in range(numcols):
        p = np.array([i, j])
        p_dash = scaling_mat.dot(p)
        new_i, new_j = p_dash[0], p_dash[1]
        gray3[i,j] = gray[i,j]


# plt.imshow(gray3, cmap="gray")
# plt.show()

displayImageInActualSize(gray3)