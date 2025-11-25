import numpy as np
import matplotlib.pyplot as plt

im = np.arange(10)
print(im.shape)
img = im[np.newaxis,:]
print(img.shape)
img_ = np.repeat(img, 100, axis=0)
print(img_.shape)
print(img_)
plt.imshow(img_, cmap='gray')
