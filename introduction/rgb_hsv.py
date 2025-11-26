import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def rgb_to_hsv(r,g,b,scale_factor = 1):

    r,g,b = r/255.0, g/255.0, b/255.0
    cmax = max(r,g,b)
    cmin = min(r,g,b)
    diff = cmax - cmin
    h = 0

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 0) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    if h<0:
         h += 360

    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * scale_factor

    v = cmax * scale_factor

    return h,s,v



if __name__ == "__main__":

    # print(rgb_to_hsv(100,200,50,100))

    img = cv.imread("images/tulips.jpg")

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lr = np.array([0,120,70])
    ur = np.array([15,255,255])

    mask1 = cv.inRange(hsv_img,lr,ur)
    
    lr2 = np.array([165,120,70])
    ur2 = np.array([180,255,255])

    mask2 = cv.inRange(hsv_img, lr2, ur2)

    mask = mask1 | mask2

    res = cv.bitwise_and(img, img, mask=mask)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1])
    plt.subplot(1,2,2)
    plt.imshow(res[:,:,::-1])
    plt.show()