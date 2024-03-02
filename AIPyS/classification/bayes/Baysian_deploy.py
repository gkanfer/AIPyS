import os
import numpy as np
import pandas as pd
import tifffile as tfi
from skimage.util import img_as_ubyte
import string
import random
import skimage
from skimage.draw import disk


from AIPyS.classification.bayes.Baysian_training import Baysian_training
from AIPyS.supportFunctions.AIPS_file_display import Compsite_display
from AIPyS.supportFunctions.AIPS_functions import deployRatioCalc,call_bin,keepObject
from AIPyS.supportFunctions.GranularityFunc import openingOperation,resize,imageToRGB


class BayesDeploy(Baysian_training):
    '''
    on the fly cell call function for activating cells

    Parameters
    ----------
    file: str
        single channel target image
    path: str
    kernel_size: int
    trace_a: int
    trace_b: int
    thold: int
        probability threshold for calling cells
    pathOut: str
    clean: int
        remove object bellow the selected area size
    saveMerge: boolean

    Returns
    -------
    binary mask for activating the called cell
    '''
    def __init__(self,kernelGran,intercept,slope,*args, **kwargs):
        self.kernelGran = kernelGran
        self.intercept = intercept
        self.slope = slope
        super().__init__(*args, **kwargs)
        self.BayesianGranularityDeploy(saveMerge=True)
    
    def id_generator(self,size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))
    
    def sdCalc(self,image,mean_value):
        squared_diff_sum = 0
        num_elements = 0
        for row in image:
            for element in row:
                squared_diff_sum += (element - mean_value) ** 2
                num_elements += 1
        return (squared_diff_sum / num_elements)**0.5
    
    
    def check_singleImage(self):
        if isinstance(self.img, list):
            assert "should be a path to single image"

    def classify(self,n):
        mu = self.intercept + self.slope * n
        prob = 1 / (1 + np.exp(-mu))
        return prob
    
    def granularityDeploy(self,table,image,mask):
        img_blank = np.zeros_like(image,np.uint8)
        table["predict"] = 0.0
        table_out =  {col: [] for col in table.columns}
        active_label = []
        imgBound = (0,np.shape(img_blank)[0],np.shape(img_blank)[1])
        for idx, (index, row) in enumerate(table.iterrows()):
            img_label = row['label'] # first cell analysis 
            try:
                stack_img, stack_mask = self.stackObjects_cellpose_ebimage_parametrs_method(image,mask,table,img_label)
            except:
                continue
            if stack_img is None:
                continue
            intensity =  np.mean(stack_img)
            #image intensity and sd before opening:
            # # video gen
            imageDecy = openingOperation(kernel = self.kernelGran, image = stack_img)
            ratio = np.mean(imageDecy)/intensity
            prob = self.classify(ratio)
            row["predict"] = prob
            if prob < self.thold:
                # saves table with low pvalue for better analysis
                for col in table.columns:
                    table_out[col].append(row[col])
            else:
                active_label.append(img_label)
                x, y = row["centroid-0"], row["centroid-1"]
                r, c = disk((int(x), int(y)), 10)
                if np.any(np.in1d(imgBound, r)) or np.any(np.in1d(imgBound, c)):
                    for col in table.columns:
                        table_out[col].append(row[col])
                else:
                    img_blank[r, c] = 255
                    for col in table.columns:
                        table_out[col].append(row[col])
        return img_blank, pd.DataFrame(table_out),active_label
      
    
    def BayesianGranularityDeploy(self,saveMerge):
        self.check_singleImage()
        pathOut = self.outPath
        image = self.img # must be single image
        mask, table = self.cellpose_segmentation(image_input = image)
        # intensity after decay
        if len(table) < 5:
            with open(os.path.join(pathOut, 'cell_count.txt'), 'r') as f:
                prev_number = f.readlines()
            new_value = int(prev_number[0]) + len(table)
            with open(os.path.join(pathOut, 'cell_count.txt'), 'w') as f:
                f.write(str(new_value))
            with open(os.path.join(pathOut, 'count.txt'), 'w') as f:
                f.write(str(len(table)))
        else:
            binary, table_sel, active_label =  self.granularityDeploy(table,image,mask)
            img_gs = img_as_ubyte(binary)
            if saveMerge:
                table_sel['predict'] = np.round(table_sel.predict.values, 2)
                #maskKeep = keepObject(table = table_sel,mask = mask)
                compsiteImage = Compsite_display(input_image=image, mask_roi=binary)
                LabeldImage = compsiteImage.display_image_label(table=table_sel, font_select="arial.ttf", font_size=14, label_draw='predict', intensity=1)
                LabeldImage.save(os.path.join(pathOut, self.id_generator() + '.png'))
            with open(os.path.join(pathOut, 'active_cell_count.txt'), 'r') as f:
                prev_number_active = f.readlines()
            new_value = int(prev_number_active[0]) + len(active_label)
            with open(os.path.join(pathOut, 'active_cell_count.txt'), 'w') as f:
                f.write(str(new_value))
            if os.path.exists(os.path.join(pathOut, 'binary.jpg')):
                os.remove(os.path.join(pathOut, 'binary.jpg'))
            skimage.io.imsave(os.path.join(pathOut, 'binary.jpg'), img_gs)
            with open(os.path.join(pathOut, 'cell_count.txt'), 'r') as f:
                prev_number = f.readlines()
            new_value = int(prev_number[0]) + len(table_sel)
            with open(os.path.join(pathOut, 'cell_count.txt'), 'w') as f:
                f.write(str(new_value))
            with open(os.path.join(pathOut, 'count.txt'), 'w') as f:
                f.write(str(len(table_sel)))
            print("done!")






