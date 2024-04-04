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


def imageToRGB(image,mask,ci =1):
    '''
    bit8 color assign 255 is very bright
    '''
    input_gs_image = image
    input_gs_image = input_gs_image*ci
    input_gs_image = (input_gs_image / input_gs_image.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = mask
    rgb_input_img[:, :, 2] = ch2_u8
    return rgb_input_img

def classify(a,b,Int, td):
    mu = a + b * Int
    prob = 1 / (1 + np.exp(-mu))
    return prob, prob > thold







