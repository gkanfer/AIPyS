import numpy as np
import string
import cv2
import pdb
import random
import skimage
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from IPython.display import clear_output
from AIPyS.supportFunctions.AIPS_file_display import Compsite_display
from AIPyS.segmentation.cellpose.AIPS_cellpose import AIPS_cellpose
from AIPyS.supportFunctions.GranularityFunc import resize

class plotObjectAreas(StackObjects_cellpose):
    def __init__(self,trainingDataPath, w, h ,*args, **kwargs):
        # require self.outPath(DataLoad) 
        self.trainingDataPath = trainingDataPath
        self.w, self.h =w,h # for resizing
        super().__init__(*args, **kwargs)
        self.areaCalc_imageGen_cp()
    
    def generate_random_string(self,length):
        letters = string.ascii_letters  # this includes both lowercase and uppercase letters
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str 

    def getMeanSD(self,image_input):
        return np.round(np.mean(image_input),3),np.round(np.std(image_input),3)
    
    def check_list(self):
        if not isinstance(self.Image_name, list):
            raise ValueError("Requries list of images path")
    
    def sdCalc(self,image,mean_value):
        squared_diff_sum = 0
        num_elements = 0
        for row in image:
            for element in row:
                squared_diff_sum += (element - mean_value) ** 2
                num_elements += 1
        return (squared_diff_sum / num_elements)**0.5
        
    
    def areaCalc_imageGen_cp(self):
        '''
        Adjusted to cellpose, Generates image per slice and a single table. for training
        Note - expects a list of images path
        Produce output table used for Bayes training (without the label option)
        and Video file with Ratio value and image file name.
        Ratio is the granular spectrum
        '''
        self.check_list()
        image_name_list = self.Image_name
        tempFolder = os.path.join(self.outPath,'AreaAnalysis')
        ImagePath = os.path.join(tempFolder,'images')
        DataPath = os.path.join(tempFolder,'data')
        os.makedirs(tempFolder, exist_ok=True)
        os.makedirs(ImagePath, exist_ok=True)
        os.makedirs(DataPath, exist_ok=True)
        dfGran = {'name':[],'intensity':[],'sd':[],'maskArea':[]}
        c = 0
        for image_path in image_name_list:
            image = skimage.io.imread(image_path)
            image_name = self.generate_random_string(7)
            # Initiate cellpose segmentation
            mask, table = self.cellpose_segmentation(image_input = image)
            # create original image:
            #print('-' * c, end = '\r')
            clear_output(wait=True)
            for idx, (index, row) in enumerate(table.iterrows()):
                if idx % 10 == 0:
                    c += 1
                    print('.'*c, end = '\r')
                img_label = row['label'] # first cell analysis 
                try:
                    stack_img, stack_mask = self.stackObjects_cellpose_ebimage_parametrs_method(image,mask,table,img_label)
                except:
                    continue
                if stack_img is None:
                    continue
                intensity =  np.mean(stack_img)
                maskArea = row['area']
                sd = self.sdCalc(stack_img,intensity)
                #image intensity and sd before opening:
                dfGran['name'].append(image_name + f'_{idx}.png')
                dfGran['intensity'].append(intensity)
                dfGran['sd'].append(sd)
                dfGran['maskArea'].append(maskArea)
                frame_obj = Compsite_display(stack_img, stack_mask, 2)
                frame = resize(image=frame_obj.rgb_out, width=self.w, hight=self.h)
                PIL_image_grey = Image.fromarray(frame).convert('L') # convery to gray scale
                PIL_image = PIL_image_grey.convert('RGB') # for getting a grey scale with red text
                draw = ImageDraw.Draw(PIL_image)
                font_size = 24
                font = ImageFont.truetype("arial.ttf", font_size)  # Adjust as necessary
                draw.text((5, 5),f'Imagename:{image_name}_{idx} \n Area: {np.round(maskArea,4)}', 'red',font=font)
                PIL_image.save(os.path.join(ImagePath,image_name + '_' + f'{idx}.png'))
        dfGran = pd.DataFrame(dfGran)
        dfGran['label'] = 0
        dfGran.to_csv(os.path.join(DataPath,'imageseq_data_area.csv'))
        
    