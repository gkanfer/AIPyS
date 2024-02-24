from AIPyS_old.segmentation.parametric.GlobalSeg import GlobalSeg
from AIPyS_old.segmentation.parametric.ImageSeqGlobSeg import ImageSeqGlobSeg
from AIPyS_old.supportFunctions.GranularityFunc import openingOperation,resize
from skimage import measure, morphology
import os
import numpy as np
import cv2 
from IPython.display import clear_output
import pdb

class GranulaityMesure(GlobalSeg):
    '''
    Note: get single tif 16bit image only
    # start_kernel = 2,end_karnel = 50, kernel_size = 20,outputTableName = None,outPath = None,
    '''
    def __init__(self,start_kernel,end_karnel, kernel_size,videoName,outputImageSize,*args, **kwargs):
        self.start_kernel = start_kernel
        self.end_karnel = end_karnel
        self.kernel_size = kernel_size
        self.videoName = videoName
        self.outputImageSize = outputImageSize
        super().__init__(*args, **kwargs)
        
        # def resize(self,image):
        #     resized = cv2.resize(image, (self.outputImageSize,self.outputImageSize), interpolation = cv2.INTER_AREA)
        #     return resized
    
    def addText(self,image_input,text):
        return cv2.putText(image_input, f'Kernel: {text[0]} \n ratio: {text[1]} \n sd: {text[2]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) #0.5 font, 1 thickness
    
    def GranularityVideo(self):
        open_vec = np.linspace(self.start_kernel, self.end_karnel, self.kernel_size, endpoint=True, dtype=int)
        image = self.img
        frame,mask = self.rgbMasking(image = image)
        # width,height,layers = frame.shape
        frame = resize(image = frame, width = self.outputImageSize,hight = self.outputImageSize) # resize(image,width,hight)
        video = cv2.VideoWriter(os.path.join(self.outPath,self.videoName), cv2.VideoWriter_fourcc(*'MJPG'),1.0,(self.outputImageSize,self.outputImageSize),  isColor = True)
        #pdb.set_trace()
        #clear_output(wait=True)
        prev_image = image
        for i,opning in enumerate(open_vec):
            openImage_curr = openingOperation(kernel = opning,image = prev_image)
            if np.sum(prev_image[mask>0]) == 0:
                denominator = np.sum(prev_image[mask>0])+0.000001
            else:
                denominator = np.sum(prev_image[mask>0])
            ratio = (np.sum(prev_image[mask>0]) - np.sum(openImage_curr[mask>0]))/denominator
            prev_image = openImage_curr
            frame = self.disply_compsite(openImage_curr, mask)
            mean = np.round(ratio,2)
            sd = np.round(np.std(openImage_curr[mask>0]),2)
            frame = resize(image = frame, width = self.outputImageSize,hight = self.outputImageSize)
            print('-' * i, end = '\r')
            # add text:
            self.addText(frame,(opning,mean,sd))
            video.write(frame)
        video.release()
    

    