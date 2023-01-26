'''
Function for AIPS DASH
'''
import xml.etree.ElementTree as xml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageEnhance
# from PIL import fromarray
import plotly.express as px
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
from skimage import io, filters, measure, color, img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
from skimage.exposure import rescale_intensity
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import base64
from datetime import datetime
import re
from utils.display_and_xml import unique_rand
from utils import display_and_xml as dx


def image_to_8bits(input):
    imType = str(input.dtype)
    if not ('uint8' in imType):
        input = img_as_ubyte(input)
    if len(np.shape(input)) < 3:
        input = input.reshape(1, np.shape(input)[0], np.shape(input)[1])
    img = input
    return img

def show_image_adjust(image, low_prec, up_prec):
    """
    image= np array 2d
    low/up precentile border of the image
    """
    percentiles = np.percentile(image, (low_prec, up_prec))
    scaled_ch1 = rescale_intensity(image, in_range=tuple(percentiles))
    return scaled_ch1
    # PIL_scaled_ch1 = Image.fromarray(np.uint16(scaled_ch1))
    # PIL_scaled_ch1.show()
    # return PIL_scaled_ch1

def px_pil_figure(img,bit,mask_name,fig_title,wh,*args):
    '''
    :param img: image input - 3 channel 8 bit image
             bit:1 np.unit16 or 2 np.unit8
             fig_title: title for display on dash
             wh: width and hight in pixels
    :return: encoded_image (e_img)
    '''
    bit = str(img.dtype)
    if bit == "bool":
        # binary
        im_pil = Image.fromarray(img)
    elif bit == "int64":
        # 16 image (normal image)
        im_pil = Image.fromarray(np.uint16(img))
    else:
        # ROI mask
        img_gs = img_as_ubyte(img)
        im_pil = Image.fromarray(img_gs)
        if len(args) > 0:
            enhancer = ImageEnhance.Contrast(im_pil)
            factor = args[0]  # gives original image
            im_pil = enhancer.enhance(factor)
    fig_ch = px.imshow(im_pil, binary_string=True, binary_backend="jpg", width=wh, height=wh,title=fig_title,binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels = False)
    fig_ch.update_layout(title_x=0.5)
    return fig_ch


def XML_creat(filename,block_size,offset,rmv_object_nuc,block_size_cyto,offset_cyto,global_ther,rmv_object_cyto,rmv_object_cyto_small):
    root = xml.Element("Segment")
    cl = xml.Element("segment") #chiled
    root.append(cl)
    block_size_ = xml.SubElement(cl,"block_size")
    block_size_.text = block_size
    offset_ = xml.SubElement(cl,"offset")
    offset_.text = "13"
    rmv_object_nuc_ = xml.SubElement(cl, "rmv_object_nuc")
    rmv_object_nuc_.text = "rmv_object_nuc"
    block_size_cyto_ = xml.SubElement(cl, "block_size_cyto")
    block_size_cyto_.text = "block_size_cyto"
    offset_cyto_ = xml.SubElement(cl, "offset_cyto")
    offset_cyto_.text = "offset_cyto"
    global_ther_ = xml.SubElement(cl, "global_ther")
    global_ther_.text = "global_ther"
    rmv_object_cyto_ = xml.SubElement(cl, "rmv_object_cyto")
    rmv_object_cyto_.text = "rmv_object_cyto"
    rmv_object_cyto_small_ = xml.SubElement(cl, "rmv_object_cyto_small")
    rmv_object_cyto_small_.text = "rmv_object_cyto_small"
    tree = xml.ElementTree(root)
    with open(filename,'wb') as f:
        tree.write(f)

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

def rgb_file_gray_scale(input_gs_image,mask=None,channel=None):
    ''' create a 3 channel rgb image from 16bit input image
        optional bin countor image from ROI image
        :parameter
        input_gs_image: 16bit nparray
        mask: 32int roi image
        channel: 0,1,2 (rgb)
        :return
        3 channel stack file 8bit image
    '''
    input_gs_image = (input_gs_image / input_gs_image.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    if mask is not None and len(np.unique(mask)) > 1:
        bin_mask = dx.binary_frame_mask(ch2_u8, mask)
        bin_mask = np.where(bin_mask == 1, True, False)
        if channel is not None:
            rgb_input_img[bin_mask > 0, channel] = 255
        else:
            rgb_input_img[bin_mask > 0, 2] = 255
    return rgb_input_img

def rgbTograyscale(input):
    '''
        create a 3 channel grayscale from np format (W,H,C)
        :parameter
        input_gs_image: 8 bit
        :return
        3 channel stack file 8bit image
    '''
    rgb_input_img = np.zeros((np.shape(input)[2], np.shape(input)[0], np.shape(input)[1]), dtype=np.uint8)
    for i in range(np.shape(input)[2]):
        rgb_input_img[i, :, :] = input[:, :, i]
    return rgb_input_img


def gray_scale_3ch(input_gs_image):
    input_gs_image = (input_gs_image / input_gs_image.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    return rgb_input_img

def plot_composite_image(img,mask,fig_title,alpha=0.2):
    # apply colors to mask
    '''
    :param img: input 3 channel grayscale image
    :param mask: mask
    :param fig_title: title
    :param alpha: transprancy for blending
    :param img_shape:
    :return:
    '''
    mask = np.array(mask, dtype=np.int32)
    mask_deci = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    cm = plt.get_cmap('CMRmap')
    colored_image = cm(mask_deci)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    #RGB pil image
    img_mask = img_as_ubyte(colored_image)
    im_mask_pil = Image.fromarray(img_mask).convert('RGB')
    img_gs = img_as_ubyte(img)
    im_pil = Image.fromarray(img_gs).convert('RGB')
    im3 = Image.blend(im_pil, im_mask_pil, alpha)
    fig_ch = px.imshow(im3, binary_string=True, binary_backend="jpg", width=700, height=700, title=fig_title,
                       binary_compression_level=0).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig_ch.update_layout(title_x=0.5)
    return fig_ch

def save_pil_to_directory(img,bit,mask_name,output_dir = 'temp',mask = None, merge_mask = None,channel=None):
    '''
    Save image composite with ROI
    :param img: image input
             bit:1 np.unit16 or 2 np.unit8
             merge_mask: ROI - ROI + 3ch greyscale input OR  -BIN -  bin + 3ch greyscale
             channel: for display Bin merge  with rgb input
    :return: encoded_image (e_img)
    '''
    bit = str(img.dtype)
    if merge_mask=='ROI':
        # img  must be 3 channel grayscale
        mask = np.array(mask, dtype=np.int32)
        mask_deci = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        cm = plt.get_cmap('CMRmap')
        colored_image = cm(mask_deci)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        # RGB pil image
        img_mask = img_as_ubyte(colored_image)
        im_mask_pil = Image.fromarray(img_mask).convert('RGB')
        img_gs = img_as_ubyte(img)
        im3 = Image.fromarray(img_gs).convert('RGB')
        im_pil = Image.blend(im3, im_mask_pil, 0.2)
    elif merge_mask=='BIN':
        #img  must be 3 channel grayscale
        img_bin = rgb_file_gray_scale(img, mask=mask, channel=channel)
        img_gs = img_as_ubyte(img_bin)
        im_pil = Image.fromarray(img_gs)
    elif merge_mask is None:
        if bit == "bool":
            # binary
            im_pil = Image.fromarray(img)
        elif bit == "int64":
            # 16 image (normal image)
            im_pil = Image.fromarray(np.uint16(img))
        else:
            # ROI mask
            img = np.uint8(img)
            roi_index_uni = np.unique(img)
            roi_index_uni = roi_index_uni[roi_index_uni > 1]
            sort_mask_buffer = np.ones((np.shape(img)[0], np.shape(img)[1], 3), dtype=np.uint8)
            for npun in roi_index_uni:
                for i in range(3):
                    sort_mask_buffer[img == npun, i] = unique_rand(2, 255, 1)[0]
            im_pil = Image.fromarray(sort_mask_buffer, mode='RGB')
    filename1 = datetime.now().strftime("%Y%m%d_%H%M%S" + mask_name)
    im_pil.save(os.path.join(output_dir, filename1 + ".png"), format='png')  # this is for image processing
    e_img = base64.b64encode(open(os.path.join('temp', filename1 + ".png"), 'rb').read())
    return e_img

def remove_gradiant_label_border(mask):
    '''
    mask from rescale return with border gradiant which is for the borders
    :return mask no borders
    '''
    mask_ = np.where(mask > 0, mask, 0)
    mask_intact = np.where(np.mod(mask_, 1) > 0, 0, mask_)
    mask_intact = np.array(mask_intact, np.uint32)
    return mask_intact


