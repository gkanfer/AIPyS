import tifffile as tfi
import skimage.measure as sme
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
# from PIL import fromarray
from numpy import asarray
from skimage import data, io
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
from skimage.morphology import disk, remove_small_objects
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank, gaussian, sobel
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import regionprops_table
from skimage.measure import perimeter
from skimage import measure
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import peak_local_max
import os
import glob
import pandas as pd
from pandas import DataFrame
from scipy.ndimage.morphology import binary_fill_holes
from skimage.viewer import ImageViewer
from skimage import img_as_float
import time
import base64
from datetime import datetime
import xml.etree.ElementTree as xml
from random import randint

def display_image_label(ch,mask,lable_draw, font_select,font_size):
    '''
    ch: 16 bit input image
    mask: mask for labale
    lable_draw: 'label' or 'area'
    font_select: copy font to the working directory ("DejaVuSans.ttf" eg)
    font_size: 4 is nice size

    return:
    info_table: table of objects measure
    PIL_image: 16 bit mask rgb of the labeled image
    '''
    # count number of objects in nuc['sort_mask']
    info_table = pd.DataFrame(
        measure.regionprops_table(
            mask,
            intensity_image=ch,
            properties=['area', 'label', 'centroid'],
        )).set_index('label')
    PIL_image = Image.fromarray(np.uint16(mask)).convert('RGB')
    #Label ROI by centroid and index
    info_table['label'] = range(2, len(info_table) + 2)
    #round
    info_table = info_table.round({'centroid-0': 0, 'centroid-1': 0})
    info_table = info_table.reset_index(drop=True)
    draw = ImageDraw.Draw(PIL_image)
    # use a bitmap font
    font = ImageFont.truetype(font_select, font_size)
    if lable_draw=='label':
        sel_lable=3
    else:
        sel_lable = 0
    for i in range(len(info_table)):
        draw.text((info_table.iloc[i, 2].astype('int64'), info_table.iloc[i, 1].astype('int64')),str(info_table.iloc[i, sel_lable]), 'red', font=font)
    return info_table, PIL_image

def outline_seg(mask,index):
    seg_mask_temp = np.zeros(np.shape(mask), dtype=np.int32)
    seg_mask_temp[mask == index] = index
    seg_mask_eros_9 = binary_erosion(seg_mask_temp, structure=np.ones((9, 9))).astype(np.float64)
    seg_mask_eros_3 = binary_erosion(seg_mask_temp, structure=np.ones((3, 3))).astype(np.float64)
    framed_mask = seg_mask_eros_3 - seg_mask_eros_9
    return framed_mask

def sum_seg(dict,mask):
    update_mask = np.zeros(np.shape(mask), dtype=np.int32)

    for item in update_mask.items():
        update_mask = update_mask + item
    return update_mask


def binary_frame_mask(ch,mask,table=None):
    '''
    Create a mask for NIS-elements to photo-activate for multiple point
    :parameter
    ch - input image
    mask - input mask (RGB) 32integer
    :return
    framed_mask (RGB)
    '''
    if table is None:
        info_table = pd.DataFrame(
            measure.regionprops_table(
                mask,
                intensity_image=ch,
                properties=['area', 'label', 'centroid'],
            )).set_index('label')
    else:
        info_table=table
    info_table['label'] = range(2, len(info_table) + 2)
    framed_mask = np.zeros(np.shape(mask), dtype=np.int32)
    if len(info_table['label']) > 1:
        for i in info_table.index.values:
            framed_mask = framed_mask + outline_seg(mask,i)
    else:
        framed_mask = outline_seg(mask,info_table.index.values)
    return framed_mask

def binary_frame_mask_single_point(mask,table=None):
    '''
    Create a mask for NIS-elements to photo-activate for single point
    :parameter
    mask - input mask (RGB)
    :return
    framed_mask (RGB)
    '''
    seg_mask_eros_9 = binary_erosion(mask, structure=np.ones((9, 9))).astype(np.float64)
    seg_mask_eros_3 = binary_erosion(mask, structure=np.ones((3, 3))).astype(np.float64)
    seg_frame = np.where(seg_mask_eros_9 + seg_mask_eros_3 == 2, 3, seg_mask_eros_3)
    framed_mask = np.where(seg_frame == 3, 0, seg_mask_eros_3)
    return framed_mask




def Centroid_map(ch, mask, mat):
    '''
        Returns center of mask map after dilation
        :parameter
        ch - Grayscale input image
        mask -
        mat - matrix for the dilation operation
        :return
        table - centroid table
        center_map -
    '''
    table = pd.DataFrame(measure.regionprops_table(mask,
                            intensity_image= ch,
                            properties=['label', 'centroid_weighted',])
                         ).set_index('label')
    center_map = np.zeros(np.shape(mask))
    y = table
    centroid_0 = np.array(y['centroid_weighted-0']).astype('int')
    centroid_1 = np.array(y['centroid_weighted-1']).astype('int')
    center_map[centroid_0, centroid_1] = 1
    center_map = binary_dilation(center_map, structure=np.ones((mat, mat))).astype(np.float64)
    return table, center_map


def seq(start, end, by=None, length_out=None):
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or n_points must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
        # Switch direction in case in start and end seems to have been switched (use sign of by to decide this behaviour)
        if start > end and by > 0:
            e = start
            start = end
            end = e
        elif start < end and by < 0:
            e = end
            end = start
            start = e
        absby = abs(by)
        if absby - width < eps:
            length_out = int(width / absby)
        else:
            # by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1)
    out = [float(start)] * length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out

def evaluate_image_output(mask):
    mask_eval = mask - 1
    mask_eval = np.where(mask_eval == 1, 0, mask_eval)
    if np.sum(mask_eval) < 3:
        mask = np.zeros(np.shape(mask_eval))
    else:
        mask = mask
    return mask

def test_image(arr):
    '''
    test whether the mask generated is empty
    :parameter
    arr - np array
    '''
    values, counts = np.unique(arr.ravel(), axis=0, return_counts=True)
    return counts

def unique_rand(inicial, limit, total):
    '''
    :parameter
    :return
    :Example
    data = unique_rand(1, 60, 6)
    print(data)
    #######################################################
            prints something like
            [34, 45, 2, 36, 25, 32]
    #######################################################
    '''
    data = []
    i = 0
    while i < total:
        number = randint(inicial, limit)
        if number not in data:
            data.append(number)
            i += 1
    return data