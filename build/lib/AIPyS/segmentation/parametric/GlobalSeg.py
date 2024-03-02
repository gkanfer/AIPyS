import numpy as np
from skimage.filters import threshold_local
from skimage.util import img_as_ubyte
from scipy.ndimage.morphology import binary_opening
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os


from AIPyS.DataLoad import DataLoad

class GlobalSeg(DataLoad):
    '''
    Initiate the Global segmentation
    Parameters
    ----------
        Image_name: (str)
        path: (str)
        block_size_cyto: int
            Detect local edges 1-99 odd
        offset_cyto: float
            Detect local edges 0.001-0.9 odd
        global_ther: float
            Percentile
        clean : int
            opening operation, matrix size must be odd
    
    Returns
    -------
        combine: img
            global threshold binary map (eg cytoplasm)
    '''
    # def __init__(self,Image_name,outPath):
    #     super().__init__(Image_name,outPath)
    def __init__(self,Image_name,outPath,block_size_cyto = 11, offset_cyto = 0.1, global_ther = 0.1, clean  = 5,channel = 2,bit8 = 255,ci = 1):
        self.block_size_cyto = block_size_cyto
        self.offset_cyto = offset_cyto
        self.global_ther = global_ther
        self.clean = clean
        self.channel = channel
        self.bit8 = bit8
        self.ci = ci
        super().__init__(Image_name,outPath)
        
    def cytosolSegmentation(self,image):
        # load matrix from seedSegmentation module
        ch2 = image
        ther_cell = threshold_local(ch2, self.block_size_cyto, "gaussian", self.offset_cyto)
        blank = np.zeros_like(ch2)
        cell_mask_1 = ch2 > ther_cell
        cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((self.clean,self.clean))).astype(np.float64)
        # global threshold
        quntile_num = np.quantile(ther_cell, self.global_ther)
        cell_mask_3 = np.where(ther_cell > quntile_num, 1, 0)
        # in-case of segmentation failed
        if np.sum(cell_mask_3)==0:
            outTarget["cell_mask_1"] = cell_mask_2
            self.out_target = outTarget
            return  self.out_target
        combine = cell_mask_2
        combine[cell_mask_3 > combine] = cell_mask_3[cell_mask_3 > combine]
        return combine

    def disply_compsite(self, image, mask):
        '''
        bit8 color assign 255 is very bright
        '''
        input_gs_image = image
        input_gs_image = input_gs_image*self.ci
        input_gs_image = (input_gs_image / input_gs_image.max()) * 255
        ch2_u8 = np.uint8(input_gs_image)
        rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
        rgb_input_img[:, :, 0] = ch2_u8
        rgb_input_img[:, :, 1] = ch2_u8
        rgb_input_img[:, :, 2] = ch2_u8
        rgb_input_img[mask > 0, self.channel] = self.bit8
        return rgb_input_img

    def rgbMasking(self,image):
        mask = self.cytosolSegmentation(image = image)
        frame = self.disply_compsite(image = image, mask = mask)
        return frame,mask

    def openingOperation_(self,kernel,image):
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
    
