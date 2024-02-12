from skimage import measure, morphology
import os
import numpy as np
import cv2

def openingOperation(kernel,image):
    '''
    Parameters
    ---------- 
        kernel: int, 
            size of filter kernel
    return
    ------
        opening operation image
    '''
    selem = morphology.disk(kernel, dtype=bool)
    eros_pix = morphology.erosion(image, footprint=selem)
    imageOpen = morphology.dilation(eros_pix, footprint=selem)
    return imageOpen


def resize(image,width,hight):
    resized = cv2.resize(image, (width,hight), interpolation = cv2.INTER_AREA)
    return resized