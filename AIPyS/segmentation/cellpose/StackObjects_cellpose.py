from AIPyS.segmentation.cellpose.AIPS_cellpose import AIPS_cellpose
import numpy as np
import skimage

class StackObjects_cellpose(AIPS_cellpose):
    '''
    function similar to the EBimage stackObjectsta, return a crop size based on center of measured mask.

    Parameters
    ----------
    extract_pixel: int
        size of extraction according to mask (e.g. 50 pixel)
    resize_pixel: int
        resize for preforming tf prediction (e.g. 150 pixel)
    img_label: int
        the mask value for stack
    Returns
    -------
    center image with out background (img)
    '''
    def __init__(self,extract_pixel, resize_pixel,*args, **kwargs):
        self.extract_pixel = extract_pixel
        self.resize_pixel = resize_pixel
        super().__init__(*args, **kwargs)
        
    
    def checkMissing(self,inputObj):
        if inputObj is None:
            raise "missing cellpose output masks tables"

    def stackObjects_cellpose_ebimage_parametrs_method(self, image_input,mask,table,img_label):
        '''
        img_label: int
            the mask value for stack
        '''
        self.checkMissing(inputObj = mask)
        self.checkMissing(inputObj = table)
        self.checkMissing(inputObj = img_label)
        #x, y = table.loc[table['label']==img_label, ["centroid-0", "centroid-1"]].values.tolist()
        #row, col = disk((int(x), int(y)), 10)
        arr = table.loc[table['label']==img_label, ["centroid-0", "centroid-1"]].values
        # #x, y = table.loc[table['label']==img_label, ["centroid-0", "centroid-1"]].values
        x, y = int(arr[:,0]), int(arr[:,1])
        mask_value = mask[x, y]
        x_start = x - self.extract_pixel
        x_end = x + self.extract_pixel
        y_start = y - self.extract_pixel
        y_end = y + self.extract_pixel
        if x_start < 0 or x_end < 0 or y_start < 0 or y_end < 0:
            stack_img = None
            mask_value = None
            return stack_img, mask_value
        else:
            mask_bin = np.zeros((np.shape(image_input)[0], np.shape(image_input)[1]), np.int32)
            mask_bin[mask == mask_value] = 1
            masked_image = image_input * mask_bin
            stack_img = masked_image[x_start:x_end, y_start:y_end]
            stack_img = skimage.transform.resize(stack_img, (self.resize_pixel, self.resize_pixel), anti_aliasing=False)
            return stack_img, mask_value