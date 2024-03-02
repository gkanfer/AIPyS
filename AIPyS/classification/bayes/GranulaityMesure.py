from AIPyS.segmentation.parametric.GlobalSeg import GlobalSeg
from AIPyS.segmentation.parametric.ImageSeqGlobSeg import ImageSeqGlobSeg
from AIPyS.supportFunctions.GranularityFunc import openingOperation,resize
from AIPyS.supportFunctions.AIPS_file_display import Compsite_display
from AIPyS.segmentation.cellpose.StackObjects_cellpose import StackObjects_cellpose
from skimage import measure, morphology
import os
import numpy as np
import cv2
import time
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

class GranulaityMesure_cp(StackObjects_cellpose):
    '''
    cellprofiler adjuste, Note: get single tif 16bit image only
    # start_kernel = 2,end_karnel = 50, kernel_size = 20,outputTableName = None,outPath = None,
    '''
    def __init__(self,videoName, start_kernel,end_karnel, kernel_size,outputImageSize,*args, **kwargs):
        self.videoName = videoName
        self.start_kernel = start_kernel
        self.end_karnel = end_karnel
        self.kernel_size = kernel_size
        self.outputImageSize = outputImageSize
        super().__init__(*args, **kwargs)
        self.GranularityVideo_cp()
        
    def addText(self,image_input,text):
        return cv2.putText(image_input, f'Kernel: {text[0]} \n ratio: {text[1]} \n sd: {text[2]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) #0.5 font, 1 thickness
    
    def check_singleImage(self):
        if isinstance(self.img, list):
            assert "should be a path to single image"
    
    def GranularityVideo_cp(self):
        '''
        randomaly chose cell id and proform opening operation on it. 
        '''
        open_vec = np.linspace(self.start_kernel, self.end_karnel, self.kernel_size, endpoint=True, dtype=int)
        self.check_singleImage()
        image = self.img # must be single image
        mask, table = self.cellpose_segmentation(image_input = image)
        # single cell randomly selected:
        img_label = table['label'].sample(n=1).iloc[0]
        stack_img, stack_mask = self.stackObjects_cellpose_ebimage_parametrs_method(image,mask,table,img_label)
        frame_obj = Compsite_display(stack_img, stack_mask, 2)
        frame = resize(image=frame_obj.rgb_out, width=self.outputImageSize, hight=self.outputImageSize)  # resize(image,width,hight)
        video = cv2.VideoWriter(os.path.join(self.outPath,self.videoName), cv2.VideoWriter_fourcc(*'MJPG'),1.0,(self.outputImageSize,self.outputImageSize),  isColor = True)
        prev_image = stack_img
        print('analysis might take a few minutes')
        for i, opning in enumerate(open_vec):
            openImage_curr = openingOperation(kernel=opning, image=prev_image)
            if np.sum(prev_image[stack_mask > 0]) == 0:
                denominator = np.sum(prev_image[stack_mask > 0]) + 0.000001
            else:
                denominator = np.sum(prev_image[stack_mask > 0])
            ratio = (np.sum(prev_image[stack_mask > 0]) - np.sum(openImage_curr[stack_mask > 0])) / denominator
            prev_image = openImage_curr
            frame_obj = Compsite_display(openImage_curr, stack_mask, 2)
            frame = resize(image=frame_obj.rgb_out, width=self.outputImageSize, hight=self.outputImageSize)
            mean = np.round(ratio, 4)
            sd = np.round(np.std(openImage_curr[stack_mask > 0]), )
            print('-' * i, end='\r')
            # add text:
            self.addText(frame, (opning, mean, sd))
            video.write(frame)
        video.release()
        print(f'{self.videoName} is saved')    