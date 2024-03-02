import numpy as np
import os
import skimage.io
import glob
import pdb

class DataLoad:
    '''
    Takes a list of image names and returns a single image or a list of images. Note: Images must be 16-bit grayscale TIFF images, expecting glob. 
    Parameters
    ----------
    Image_name: str 
        glob module function
    outpath: str
    '''
    def __init__(self, Image_name = None, outPath = None):
        self.Image_name = Image_name
        self.outPath = outPath
        self.img = self.imageLoading()
    
    # def check_path(self,input):
    #     assert os.path.dirname(input), "The file_name object does not contain a path."
    
    def multiImageLoading(self):
        img = []
        for image in self.Image_name:
            img.append(self.singleimageLoading(imageName = image))
        return img
    
    def singleimageLoading(self,imageName):
        return skimage.io.imread(imageName)
    
    def imageLoading(self):
        # self.check_path(self.Image_name)
        # self.check_path(self.outPath)
        if isinstance(self.Image_name, list): 
            return self.multiImageLoading()
        else:
            return self.singleimageLoading(imageName = self.Image_name)
            