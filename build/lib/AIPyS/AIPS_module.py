import tifffile as tfi
import numpy as np
from skimage.filters import threshold_local
from skimage.util import img_as_ubyte
from scipy.ndimage.morphology import binary_opening,binary_erosion
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from utils.display_and_xml import evaluate_image_output,test_image
from utils.AIPS_functions import rgbTograyscale

class AIPS():
    '''
    Initiate the AIPS object
    '''
    def __init__(self, Image_name=None, path=None, inputImg = None):
        self.Image_name = Image_name
        self.path = path
        self.inputImg = inputImg
        if Image_name and path:
            input = tfi.imread(os.path.join(self.path, self.Image_name))
            imType = str(input.dtype)
            if not ('uint8' in imType):
                input = img_as_ubyte(input)
            if len(np.shape(input)) < 3:
                input = input.reshape(1, np.shape(input)[0], np.shape(input)[1])
            # stack channel is last
            elif np.shape(input)[0] > 10:
                input = rgbTograyscale(input)
            self.inputImg = input

    def imageMatrix(self):
        return self.inputImg

class Segmentation(AIPS):
    def __init__(self,Image_name,path,ch_=None,rmv_object_nuc=None, block_size=None, offset=None, clean = None, seed_out = None, out_target = None):
        super().__init__(Image_name,path)
        self.ch_ = ch_
        self.rmv_object_nuc = rmv_object_nuc
        self.block_size = block_size
        self.offset = offset
        self.clean = clean
        self.seed_out = seed_out
        self.out_target = out_target
    def seedSegmentation(self):
        '''
        ch_: Channel selected
        block_size: Detect local edges 1-99 odd
        offset: Detect local edges 0.001-0.9 odd
        rmv_object_nuc: percentile of cells to remove, 0.01-0.99
        clean: opening operation, matrix size must be odd
        :return:
        nmask2: local threshold binary map (eg nucleus)
        nmask4: local threshold binary map post opening (eg nucleus)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        sort_mask_bin: Binary
        '''
        #
        ch = self.inputImg[self.ch_]
        nmask = threshold_local(ch, self.block_size, "mean", self.offset)
        blank = np.zeros_like(ch)
        # empty dictionary
        out = {'nmask2': blank, 'nmask4': blank, 'sort_mask': blank,
                'tabale_init': None, 'table': None}
        nmask2 = ch > nmask
        if self.clean % 2 == 0:
            out['nmask2'] = nmask2
            self.seed_out = out
            return self.seed_out
        # remove small object according to the matrix size store in clean variable
        nmask3 = binary_opening(nmask2, structure=np.ones((self.clean,self.clean))).astype(np.uint8)
        nmask4 = binary_fill_holes(nmask3)
        # Bin to ROI
        label_objects = sm.label(nmask4, background=0)
        # Calculate object size
        info_table = pd.DataFrame(
            measure.regionprops_table(
                label_objects,
                intensity_image=ch,
                properties=['area', 'label', 'coords', 'centroid'],
            )).set_index('label')
        table_init = info_table
        # remove size by quantile input
        test = info_table[info_table['area'] < info_table['area'].quantile(q=self.rmv_object_nuc)]
        sort_mask = label_objects
        if len(test) > 0:
            x = np.concatenate(np.array(test['coords']))
            sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
            sort_mask_bin = np.where(sort_mask > 0, 1, 0)
            out = {'nmask2': nmask2, 'nmask4': nmask4, 'sort_mask': sort_mask,
                    'sort_mask_bin': sort_mask_bin, 'tabale_init': table_init, 'table': test}
            self.seed_out = out
            return self.seed_out
        else:
            out = {'nmask2': nmask2, 'nmask4': nmask4, 'sort_mask': blank,
                    'sort_mask_bin': blank,'tabale_init': table_init, 'table': test}
            self.seed_out = out
            return self.seed_out

    def cytosolSegmentation(self, ch2_,block_size_cyto,offset_cyto,global_ther,rmv_object_cyto,rmv_object_cyto_small,remove_borders=False):
        '''
        ch: Input image (tifffile image object)
        ch2: Input image (tifffile image object)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        sort_mask_bin: Binary
        block_size_cyto: Detect local edges 1-99 odd
        offset_cyto: Detect local edges 0.001-0.9 odd
        global_ther: Percentile
        rmv_object_cyto:  percentile of cells to remove, 0.01-0.99
        rmv_object_cyto_small:  percentile of cells to remove, 0.01-0.99
        remove_border: boolean -  object on border of image to be removed
        :return:
        nmask2: local threshold binary map (eg nucleus)
        nmask4: local threshold binary map post opening (eg nucleus)
        sort_mask: RGB segmented image output first channel for mask (eg nucleus)
        cell_mask_2: local threshold binary map (eg cytoplasm)
        combine: global threshold binary map (eg cytoplasm)
        sort_mask_syn: RGB segmented image output first channel for mask (eg nucleus) sync
        mask_unfiltered: Mask before filtering object size
        cseg_mask: RGB segmented image output first channel for mask (eg nucleus)
        cseg_mask_bin: Binary mask
        test: Area table seed
        info_table: Area table cytosol synchronize
        table_unfiltered: Table before remove large and small objects
        '''
        # load matrix from seedSegmentation module
        sort_mask = self.seed_out['sort_mask']
        sort_mask_bin = self.seed_out['sort_mask_bin']
        # gaussian threshold
        ch2 = self.inputImg[ch2_]
        ther_cell = threshold_local(ch2, block_size_cyto, "gaussian", offset_cyto)
        blank = np.zeros_like(ch2)
        # initiate output
        outTarget = {'cell_mask_1': blank,'combine': blank,'sort_mask_sync':blank,'mask_unfiltered':blank,'cseg_mask': blank,'cseg_mask_bin':blank,'info_table': None,'table_unfiltered':None}
        cell_mask_1 = ch2 > ther_cell
        cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((self.clean,self.clean))).astype(np.float64)
        # global threshold
        quntile_num = np.quantile(ther_cell, global_ther)
        cell_mask_3 = np.where(ther_cell > quntile_num, 1, 0)
        # in-case of segmentation failed
        if np.sum(cell_mask_3)==0:
            outTarget["cell_mask_1"] = cell_mask_2
            self.out_target = outTarget
            return  self.out_target
        combine = cell_mask_2
        combine[cell_mask_3 > combine] = cell_mask_3[cell_mask_3 > combine]
        combine[sort_mask_bin > combine] = sort_mask_bin[sort_mask_bin > combine]
        # using seedSegmentation module matrix as seed and watershed for outline
        cseg = watershed(np.ones_like(sort_mask_bin), sort_mask, mask=cell_mask_2)
        csegg = watershed(np.ones_like(sort_mask), cseg, mask=combine)
        info_table = pd.DataFrame(
            measure.regionprops_table(
                csegg,
                intensity_image=ch2,
                properties=['area', 'label', 'centroid','coords'],
            )).set_index('label')
        ############# remove large object ################
        cseg_mask = csegg
        table_unfiltered = info_table
        test1 = info_table[info_table['area'] > info_table['area'].quantile(q = rmv_object_cyto)]
        if len(test1) > 0:
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            cseg_mask = cseg_mask
        ############# remove small object ################
        test2 = info_table[info_table['area'] < info_table['area'].quantile(q = rmv_object_cyto_small)]
        if len(test2) > 0:
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            cseg_mask = cseg_mask
        # sync seed mask with cytosol mask
        if remove_borders:
            y_axis = np.shape(ch2)[0]
            x_axis = np.shape(ch2)[1]
            empty_array = np.zeros(np.shape(ch2))
            empty_array[0:1, 0:y_axis] = cseg_mask[0:1, 0:y_axis]  # UP
            empty_array[y_axis - 1:y_axis, 0:y_axis] = cseg_mask[y_axis - 1:y_axis, 0:y_axis]  # down
            empty_array[0:x_axis, 0:1] = cseg_mask[0:x_axis, 0:1]
            empty_array[0:x_axis, y_axis - 1:y_axis] = cseg_mask[0:x_axis, y_axis - 1:y_axis]  # left
            u, indices = np.unique(empty_array[empty_array > 0], return_inverse=True) #u is unique values greater then zero
            remove_border_ = list(np.int16(u))
            for i in list(remove_border_):
                cseg_mask = np.where(cseg_mask == i, 0, cseg_mask)
            info_table = pd.DataFrame(
                measure.regionprops_table(
                    cseg_mask,
                    intensity_image=ch2,
                    properties=['area', 'label', 'centroid'],
                )).set_index('label')
        else :
            if len(info_table) > 1:
                info_table = pd.DataFrame(
                    measure.regionprops_table(
                        cseg_mask,
                        intensity_image=ch2,
                        properties=['area', 'label', 'centroid'],
                    )).set_index('label')
            else:
                self.out_target = outTarget
                return self.out_target
        len_unfiltered = len(table_unfiltered)
        len_test1 = len(table_unfiltered.drop(test1.index))
        len_test2 = len(table_unfiltered.drop(test2.index))
        dict_object_table = {'Start':len_unfiltered,"remove large objects":len_test1,"remove small objects":len_test2}
        table_object_summary = pd.DataFrame(dict_object_table,index=[0])
        # set mask for seed and target
        sort_mask_bin = np.where(sort_mask > 0, 1, 0)
        cseg_mask_bin = np.where(cseg_mask > 0, 1, 0)
        combine_namsk = np.where(sort_mask_bin + cseg_mask_bin > 1, sort_mask, 0)
        # dictionary update
        outTarget['combine'] = combine
        outTarget['sort_mask_sync'] = combine_namsk
        outTarget['mask_unfiltered'] = cseg
        outTarget['cseg_mask'] = cseg_mask
        outTarget['cseg_mask_bin'] = cseg_mask_bin
        outTarget['info_table'] = info_table
        outTarget['table_unfiltered'] = table_unfiltered
        self.out_target = outTarget
        return self.out_target