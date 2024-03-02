import numpy as np
from skimage.util import img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
import os
import cv2
from AIPyS.supportFunctions.AIPS_file_display import Compsite_display
from AIPyS.segmentation.cellpose.AIPS_cellpose import AIPS_cellpose
from AIPyS.supportFunctions.GranularityFunc import resize

class CellPoseSeg(AIPS_cellpose):
    def __init__(self,videoName,*args, **kwargs):
        self.videoName = videoName
        super().__init__(*args, **kwargs)
        self.imageSeqGen_cp()
    
    def check_multiImage(self):
        if len(self.img) < 2:
            assert "list of images are required"
    
    def imageSeqGen_cp(self):
        self.check_multiImage() # except list of images
        images = self.img
        mask, _ = self.cellpose_segmentation(image_input = images[0])
        frame_obj = Compsite_display(input_image = images[0], mask_roi = mask, channel =  2)
        frame = resize(frame_obj.rgb_out, 500,500)
        width, height, layers = frame.shape
        video = cv2.VideoWriter(os.path.join(self.outPath, self.videoName), cv2.VideoWriter_fourcc(*'MJPG'), 1.0,(500, 500), isColor=True)
        print('analysis might take a few minutes')
        for i, image in enumerate(images):
            print('-' * i, end='\r')
            mask, _ = self.cellpose_segmentation(image_input=image)
            frame_obj = Compsite_display(input_image = image, mask_roi = mask, channel =  2)
            frame = resize(frame_obj.rgb_out, 500,500)
            video.write(frame)
        video.release()
        print(f'{self.videoName} is saved')






