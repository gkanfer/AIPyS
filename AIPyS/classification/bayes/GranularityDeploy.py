from AIPyS_old.classification.bayes.GranularityDataGen import GranularityDataGen
from AIPyS_old.supportFunctions.GranularityFunc import openingOperation
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

class GranularityDeploy(GranularityDataGen):
    '''
    This function is designed to analyze an image by employing a sliding window approach. As input, it accepts an image and systematically traverses it with a running window. For each segment captured within this window, the function utilizes a Bayesian model to predict the class to which the segment belongs. Upon completing the traversal, it generates a comprehensive table. This table showcases the predictions for each slice, alongside the precise coordinates. These coordinates are instrumental in selecting and referencing the relevant entries in the class prediction table, thereby facilitating a detailed and organized analysis of the image based on the Bayesian classification.
    '''
    def __init__(self,a,b,td,expansion_factor,*args, **kwargs):
        self.a = a #intercept
        self.b = b #slope
        self.td = td
        self.expansion_factor = expansion_factor
        super().__init__(*args, **kwargs)

    def classify(self,Int):
        mu = self.a + self.b * Int
        prob = 1 / (1 + np.exp(-mu))
        return prob

    def creatCrudeMask(self,img_curr,x_start,y_start,winSize):
        height, width = img_curr.shape[:2]
        center_x, center_y = x_start + winSize // 2, y_start + winSize // 2
        new_winSize = int(winSize * self.expansion_factor)
        center_x, center_y = x_start + winSize // 2, y_start + winSize // 2
         # Adjusting new window coordinates based on the image boundaries
        new_x_start = max(center_x - new_winSize // 2, 0)
        new_y_start = max(center_y - new_winSize // 2, 0)
        new_x_end = min(center_x + new_winSize // 2, width)
        new_y_end = min(center_y + new_winSize // 2, height)
    
        # Handling cases where the window hits the boundary and keeping the aspect ratio
        if new_x_end - new_x_start < new_winSize:
            diff = new_winSize - (new_x_end - new_x_start)
            new_x_start = max(new_x_start - diff, 0)
            new_x_end = min(new_x_end + diff, width)
        if new_y_end - new_y_start < new_winSize:
            diff = new_winSize - (new_y_end - new_y_start)
            new_y_start = max(new_y_start - diff, 0)
            new_y_end = min(new_y_end + diff, height)
        new_x_start = max(0, new_x_start)
        new_y_start = max(0, new_y_start)
        new_x_end = min(new_x_end, width)
        new_y_end = min(new_y_end, height)
        return (new_x_start, new_y_start, new_x_end, new_y_end)

    
    def predictCrop(self,savedict = False):
        img = self.img # should be single image
        mask = self.cytosolSegmentation(img)
        img_crudeMasked = np.zeros_like(img) # container for masked image
        dataContainer = {'x':[],'y':[],'ratio':[],'pred':[]}
        if not isinstance(self.windowSize, (int, float)):
                raise ValueError("window size is missing")
        if not isinstance(self.rnadomWindows, (int, float)):
            raise ValueError("rando crop generator is missing choose a number")
        minAxsis = min(img.shape[0],img.shape[1])
        winSize = self.windowSize
        while minAxsis % winSize != 0:
            minAxsis -= 1
        stepSize =  int(minAxsis/winSize)
        print(f'max image size: {minAxsis}, step size: {stepSize}')
        for x in range(0,minAxsis,stepSize):
            for y in range(0,minAxsis,stepSize):
                imageCrop = img[x:x+stepSize,y:y+stepSize]
                maskCrop = mask[x:x+stepSize,y:y+stepSize]
                imageDecy = openingOperation(kernel = self.kernelGran, image = imageCrop)
                if np.sum(imageDecy[maskCrop>0]) == 0:
                    denominator = np.sum(imageCrop[maskCrop > 0]) + 0.000001
                else:
                    denominator = np.sum(imageCrop[maskCrop > 0])
                ratio = np.sum(imageCrop[maskCrop > 0] - imageDecy[maskCrop > 0])/denominator
                pred = self.classify(ratio)
                if savedict:
                    dataContainer['ratio'].append(ratio)
                    dataContainer['pred'].append(pred)
                    dataContainer['x'].append(x)
                    dataContainer['y'].append(y)
                else:
                    # preform segmentation on the fly
                    if pred > self.td:
                        xs,ys,xe,ye = self.creatCrudeMask(imageCrop,x,y,winSize = stepSize)
                        img_crudeMasked[xs:xe,ys:ye] = img[xs:xe,ys:ye] #insert selected cell 
        if savedict:
            return dataContainer
        else:
            return img_crudeMasked


    