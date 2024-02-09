from AIPyS.segmentation.parametric.GlobalSeg import GlobalSeg
import os
import numpy as np
import cv2 
from IPython.display import clear_output
import pdb
from PIL import Image, ImageDraw,ImageFont

class ImageSeqGlobSeg(GlobalSeg):
    '''
    Image sequences of segmented files are generated and saved as global segmented files, stored in the MP4 format. 
    These are expected to be 8-bit images with the shape (w, h, c), where 'c' represents 3 channels.
    '''
    # def __init__(self,videoName,Image_name,outPath,block_size_cyto, offset_cyto, global_ther, clean, channel, bit8, ci):
    #     self.videoName = videoName #.mp4
    #     super().__init__(Image_name,outPath,block_size_cyto, offset_cyto, global_ther, clean, channel, bit8, ci)

    def __init__(self,videoName,*args, **kwargs):
        self.videoName = videoName #.mp4
        super().__init__(*args, **kwargs)
        
    def check_multiImage(self):
        if len(self.img) < 2:
            assert "list of images are required"

    # def rgbMasking(self,image):
    #     mask = self.cytosolSegmentation(image = image)
    #     frame = self.disply_compsite(image = image, mask = mask)
    #     return frame,mask
    
    def addText(self,image_input,text):
        return cv2.putText(image_input, f'mean: {text[0]} \n sd: {text[1]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    def resize(self,image):
        resized = cv2.resize(image, (500,500), interpolation = cv2.INTER_AREA)
        return resized
        
    def imageSeqGen(self):
        self.check_multiImage()
        images = self.img
        #pdb.set_trace()
        frame,mask = self.rgbMasking(image = images[0])
        width,height,layers = frame.shape
        frame = self.resize(frame)
        video = cv2.VideoWriter(os.path.join(self.outPath,self.videoName), cv2.VideoWriter_fourcc(*'MJPG'),1.0,(500,500),  isColor = True)
        #clear_output(wait=True)
        for i, image in enumerate(images):
            mean,sd = np.round(np.mean(image),2), np.round(np.std(image),2)
            frame,mask = self.rgbMasking(image = image)
            # add text:
            print('-' * i, end = '\r')
            #pilimage = Image.fromarray(frame)
            frame = self.resize(frame)
            self.addText(frame,(mean,sd))
            video.write(frame)
        video.release()
        