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
        



