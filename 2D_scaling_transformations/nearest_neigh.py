import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def f_bilinearInterpolate(r,c,I):
    lc = int(c)
    rc = lc+1
    wr = c - lc #right weight
    wl = rc - c #left weight
    tr = int(r)
    br = tr+1
    wt = br-r #top weight
    wb = r-tr #bottom weight
    if tr >= 0 and br < I.shape[0] and lc >=0 and rc < I.shape[1]:
        a = wl*I[tr,lc] + wr*I[tr,rc]
        b = wl*I[br,lc] + wr*I[br,rc]
        g = wt*a + wb*b
        return np.uint8(g)
    else:
        return 0

grayimg = "../introduction/images/albert-einstein_gray.jpg"
gray = cv.imread(grayimg, cv.IMREAD_GRAYSCALE)

numrows = gray.shape[0]
numcols = gray.shape[1]

sx, sy = 2,2
scaling_mat = np.array([[sx,0],[0,sy]])
s_inv = np.linalg.inv(scaling_mat)

new_img = np.zeros((numrows*2, numcols*2), dtype='uint8')

for new_i in range(new_img.shape[0]):

    for new_j in range(new_img.shape[1]):

        p_dash = np.array([new_i, new_j])
        p = s_inv.dot(p_dash)
        p = np.int16(np.floor(p))
        i, j = p[0], p[1]
        new_img[new_i,new_j] = gray[i,j]

plt.imshow(new_img, cmap="gray")
plt.show()
