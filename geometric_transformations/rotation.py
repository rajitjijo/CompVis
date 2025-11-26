import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def f_bilinearInterpolate(r,c,I):
    lc = int(c)
    rc = lc+1
    wr = c - lc
    wl = rc - c
    tr = int(r)
    br = tr+1
    wt = br-r
    wb = r-tr
    if tr >= 0 and br < I.shape[0] and lc >=0 and rc < I.shape[1]:
        a = wl*I[tr,lc] + wr*I[tr,rc]
        b = wl*I[br,lc] + wr*I[br,rc]
        g = wt*a + wb*b
        return np.uint8(g)
    else:
        return 0

def get_extents(R, rmax, cmax):

    coords = np.array([[0,0], [0,cmax-1], [rmax-1,0], [rmax-1,cmax-1]])
    rotated_coords = R.dot(coords.T)
    mins = rotated_coords.min(axis=1)
    maxs = rotated_coords.max(axis=1)
    minR = np.int64(np.floor(mins[0]))
    minC = np.int64(np.floor(mins[1]))
    maxR = np.int64(np.ceil(maxs[0]))
    maxC = np.int64(np.ceil(maxs[1]))
    H = (maxR - minR) + 1
    W = (maxC - minC) + 1

    return minR, minC, maxR, maxC, H, W

def f_transform(R, gray):

    rmax, cmax = gray.shape[0], gray.shape[1]
    minR, minC, maxR, maxC, H, W = get_extents(R, rmax, cmax)
    new_I = np.zeros((H,W), dtype="uint8")
    Rinv = np.linalg.inv(R)
    for new_i in range(minR,maxR):
        for new_j in range(minC,maxC):
            P_dash = np.array([new_i,new_j])
            P = Rinv.dot(P_dash)
            i , j = P[0] , P[1]
            if i < 0 or i>=rmax or j<0 or j>=cmax:
                pass
            else:
                g = f_bilinearInterpolate(i,j,gray)
                new_I[new_i-minR,new_j-minC] = g

    return new_I



if __name__ == "__main__":

    grayimg = "../introduction/images/albert-einstein_gray.jpg"
    gray = cv.imread(grayimg, cv.IMREAD_GRAYSCALE)
    a = 90
    ca = np.cos(np.deg2rad(a))
    sa = np.sin(np.deg2rad(a))
    R = np.array([[ca,-sa],[sa,ca]])

    new_img = f_transform(R, gray)
    plt.imshow(new_img, cmap='gray')
    plt.show()

