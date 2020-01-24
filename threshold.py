# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 06:09:52 2020

@author: sowme
"""

import scipy
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import imageio

grayscale = cv2.imread('out.png')
median_filtered = scipy.ndimage.median_filter(grayscale, size=3)

threshold = skimage.filters.threshold_otsu(median_filtered)

print('Threshold value is {}'.format(threshold))
predicted = np.uint8(median_filtered > threshold) * 255

plt.imshow(predicted, cmap='gray')
plt.axis('off')
plt.title('otsu predicted binary image')
imageio.imwrite('thres.png', predicted)
