from __future__ import print_function
import os                                    
import sys
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

busterNet_root = 'BusterNet/'
busterNet_srcDir = os.path.join( busterNet_root, 'Model' )
sys.path.insert( 0, busterNet_srcDir )

from BusterNet.Model.BusterNetCore import create_BusterNet_testing_model
busterNetModel = create_BusterNet_testing_model( 'BusterNet/Model/pretrained_busterNet.hd5' )

from BusterNet.Model.BusterNetUtils import *
import cv2
from skimage.color import rgb2gray
from skimage.color import gray2rgb

rgb = cv2.imread('beach-wood-copy.png')
pred = simple_cmfd_decoder( busterNetModel, rgb )

def draw_image(img, title, height, width):
    plt.figure(figsize = (height,width))  
    plt.title(title)
    imgplot = plt.imshow(img, 'gray')
    
grayscale = rgb2gray(pred)
draw_image(grayscale, "Grayscale", 5, 5)
draw_image(rgb, "RGB", 5, 5)