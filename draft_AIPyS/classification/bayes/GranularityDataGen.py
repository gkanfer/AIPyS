from AIPyS.classification.bayes.RunningWindow import RunningWindow
from AIPyS.supportFunctions.GranularityFunc import openingOperation,resize
import numpy as np
import pandas
import string
import cv2
import pdb
import random
import skimage
import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from IPython.display import clear_output

class GranularityDataGen(RunningWindow):
    def __init__(self,kernelGran,w, h ,*args, **kwargs):
        # require self.outPath(DataLoad) 
        self.kernelGran = kernelGran # The kernelGran is dictated by the GranularityVideo generated video file of Granularity process (example can be 6)
        self.w, self.h =w,h # for resizing
        super().__init__(*args, **kwargs)

    def addText_splitVideo(self,image_input,text):
        return cv2.putText(image_input, f'Imagename: {text[0]} \n Ratio: {text[1]} ', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) #0.5 font, 1 thickness
    
    
    def generate_random_string(self,length):
        letters = string.ascii_letters  # this includes both lowercase and uppercase letters
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str 

    def imageMeasure(self,inage_input, mask):
        return np.mean(inage_input[mask>0]), np.std(inage_input[mask>0])
      
    def getMeanSD(self,input_image):
        return np.round(np.mean(input_image),3),np.round(np.std(input_image),3)

    def granCalc(self):
        '''
        Produce output table used for Bayes training (without the label option)
        and Video file with Ratio value and image file name.
        Ratio is the granular spectrum
        '''
        # pdb.set_trace()
        image_name = self.generate_random_string(7)
        tempFolder = os.path.join(self.outPath,image_name)
        ImageOrig = os.path.join(tempFolder,'ImageOrig')
        ImageSplit = os.path.join(tempFolder,'ImageSplit')
        os.makedirs(tempFolder,exist_ok=True)
        os.makedirs(ImageOrig,exist_ok=True)
        os.makedirs(ImageSplit,exist_ok=True)
        image = self.img # should be single image
        mask = self.cytosolSegmentation(image)
        # creat original image:
        mean,sd = self.imageMeasure(image, mask)
        skimage.io.imsave(os.path.join(ImageOrig,image_name + f'_mean_{mean}' + f'_sd_{sd}' '.tif'), image)
        imageSlice,maskSlice = self.runningWin(image_input = image, mask = mask)
        if imageSlice is None or maskSlice is None:
            raise ValueError('None is not a valid list')
        dfGran = {'name':[],'ratio':[],'intensity':[],'sd':[]}
        
        open_vec = np.linspace(2,self.kernelGran, self.kernelGran, endpoint=True, dtype=int).tolist()
        open_vec = list(set([open for open in open_vec if open % 2 ==0 ]))
        # video gen
        open_vec = np.linspace(2, self.kernelGran, self.kernelGran, endpoint=True, dtype=int)
        frame = self.disply_compsite(imageSlice[0], maskSlice[0])
        width,height = self.w,self.h
        video = cv2.VideoWriter(os.path.join(ImageSplit,image_name + '.avi'), cv2.VideoWriter_fourcc(*'MJPG'),1.0,(width,height),  isColor = True)
        for i,(imageCrop,maskCrop) in enumerate(zip(imageSlice,maskSlice)):
            prev_image = imageCrop
            for open in open_vec:
                imageDecy = openingOperation(kernel = open, image = prev_image)
                if np.sum(imageDecy[maskCrop>0]) == 0:
                    denominator = np.sum(prev_image[maskCrop > 0]) + 0.000001
                else:
                    denominator = np.sum(prev_image[maskCrop > 0])
                ratio = np.sum(prev_image[maskCrop > 0] - imageDecy[maskCrop > 0])/denominator
                prev_image = imageDecy
            mean,sd = self.imageMeasure(imageDecy, maskCrop)
            dfGran['name'].append(image_name + f'_i')
            dfGran['ratio'].append(ratio)
            dfGran['intensity'].append(mean)
            dfGran['sd'].append(sd)
            frame = self.disply_compsite(imageCrop, maskCrop)
            frame = resize(frame, width ,height)
            print('-' * i, end = '\r')
            self.addText_splitVideo(frame,(image_name + f'_{i}',np.round(ratio,4)))
            video.write(frame)
        video.release()
        dfGran = pd.DataFrame(dfGran)
        dfGran['label'] = 0
        dfGran.to_csv(os.path.join(ImageSplit,image_name + '.csv'))
        
    def check_list(self):
        if not isinstance(self.Image_name, list):
            raise ValueError("Requries list of images path")
    
    def granCalc_imageGen(self):
        '''
        Generates image per slice and a single table.
        Note - expects a list of images path
        Produce output table used for Bayes training (without the label option)
        and Video file with Ratio value and image file name.
        Ratio is the granular spectrum
        '''
        # pdb.set_trace()
        self.check_list()
        image_name_list = self.Image_name
        tempFolder = os.path.join(self.outPath,'imageSequence')
        ImagePath = os.path.join(tempFolder,'images')
        ImageOrig = os.path.join(tempFolder,'ImageOrig')
        DataPath = os.path.join(tempFolder,'data')
        os.makedirs(tempFolder, exist_ok=True)
        os.makedirs(ImagePath, exist_ok=True)
        os.makedirs(ImageOrig,exist_ok=True)
        os.makedirs(DataPath, exist_ok=True)
        dfGran = {'name':[],'ratio':[],'intensity':[],'sd':[],'maskArea':[]}
        dfGranImageTotal = {'name':[],'intensity':[],'sd':[],'maskArea':[]}
        for image_path in image_name_list:
            image = skimage.io.imread(image_path)
            image_name = self.generate_random_string(7)
            mask = self.cytosolSegmentation(image)
            # creat original image:
            mean,sd = self.imageMeasure(image, mask)
            maskArea = np.sum(mask>0)/np.sum(np.ones_like(mask))
            #image intensity and sd before slicing:
            dfGranImageTotal['name'].append(image_name + f'.png')
            dfGranImageTotal['intensity'].append(mean)
            dfGranImageTotal['sd'].append(sd)
            dfGranImageTotal['maskArea'].append(maskArea)
            skimage.io.imsave(os.path.join(ImageOrig,image_name + f'_mean_{mean}' + f'_sd_{sd}' '.tif'), image)
            imageSlice,maskSlice = self.runningWin(image_input = image, mask = mask)
            if imageSlice is None or maskSlice is None:
                raise ValueError('None is not a valid list')
            open_vec = np.linspace(2,self.kernelGran, self.kernelGran, endpoint=True, dtype=int).tolist()
            open_vec = list(set([open for open in open_vec if open % 2 ==0 ]))
            # video gen
            open_vec = np.linspace(2, self.kernelGran, self.kernelGran, endpoint=True, dtype=int)
            for i,(imageCrop,maskCrop) in enumerate(zip(imageSlice,maskSlice)):
                clear_output(wait=True)
                prev_image = imageCrop
                maskArea = np.sum(maskCrop>0)/np.sum(np.ones_like(maskCrop))
                if maskArea < 0.4:
                    continue
                else:
                    for open in open_vec:
                        imageDecy = openingOperation(kernel = open, image = prev_image)
                        if np.sum(imageDecy[maskCrop>0]) == 0:
                            denominator = np.sum(prev_image[maskCrop > 0]) + 0.000001
                        else:
                            denominator = np.sum(prev_image[maskCrop > 0])
                        ratio = np.sum(prev_image[maskCrop > 0] - imageDecy[maskCrop > 0])/denominator
                        prev_image = imageDecy
                    mean,sd = self.imageMeasure(imageDecy, maskCrop)
                    dfGran['name'].append(image_name + f'_{i}.png')
                    dfGran['ratio'].append(ratio)
                    dfGran['intensity'].append(mean)
                    dfGran['sd'].append(sd)
                    dfGran['maskArea'].append(maskArea)
                    frame = self.disply_compsite(imageCrop, maskCrop)
                    frame = resize(frame, 500 ,500)
                    PIL_image_grey = Image.fromarray(frame).convert('L') # convery to gray scale
                    PIL_image = PIL_image_grey.convert('RGB') # for getting a grey scale with red text
                    draw = ImageDraw.Draw(PIL_image)
                    font_size = 24
                    font = ImageFont.truetype("arial.ttf", font_size)  # Adjust as necessary
                    draw.text((5, 5),f'Imagename:{image_name}_{i} \n Ratio:{np.round(ratio,2)} \n Area: {np.round(maskArea,2)}', 'red',font=font)
                    print('-' * i, end = '\r')
                    PIL_image.save(os.path.join(ImagePath,image_name + '_' + f'{i}.png'))
        dfGran = pd.DataFrame(dfGran)
        dfGran['label'] = 0
        dfGran.to_csv(os.path.join(DataPath,'imageseq_data.csv'))
        dfGranImageTotal = pd.DataFrame(dfGranImageTotal)
        dfGranImageTotal.to_csv(os.path.join(DataPath,'imageOrig_data.csv'))
        


