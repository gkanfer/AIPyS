import numpy as np
import os
import skimage.io
import matplotlib.pyplot as plt
from cellpose import models, core
import pandas as pd
from scipy.stats import skew

from PIL import Image, ImageDraw,ImageFont
from skimage import img_as_ubyte
from skimage.draw import disk
from skimage import measure

import seaborn as sns
import time
from AIPyS.DataLoad import DataLoad

class AIPS_cellpose(DataLoad):
    """
    Cellpose algorithm

    Parameters
    ----------
    model_type: str
        'cyto' or model_type='nuclei'
    clean: int
        remove object bellow the selected area size
    channels: list
        channels = [0,0] # IF YOU HAVE GRAYSCALE
        channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
        channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
        or if you have different types of channels in each image
        channels = [[2,3], [0,0], [0,0]]
        channels = [1,1]
    diameter: int
        e.g. 50 for 20X objective
    """
    def __init__(self,model_type,channels,diameter,*args, **kwargs):
        self.model_type = model_type
        self.channels = channels
        self.diameter = diameter
        super().__init__(*args, **kwargs)
        
    def cellpose_segmentation(self, image_input):
        '''
        returns a binery mask (np array 16bit) and a table(coloumns: area	label	centroid-0	centroid-1)
        '''
        if image_input is None:
            image_input = self.img # should be single greyscale image
        use_GPU = core.use_gpu()
        model = models.Cellpose(gpu=use_GPU, model_type=self.model_type)
        start_time = time.time()
        mask, _, _, _ = model.eval(image_input, diameter=self.diameter, flow_threshold=None, channels=self.channels)
        table = pd.DataFrame(
            measure.regionprops_table(
                mask,
                intensity_image=image_input,
                properties=['area', 'label', 'centroid','intensity_mean']))
        end_time = time.time()  # Record the end time
        duration = end_time - start_time  # Calculate the duration
        if duration > 60:  # 60 seconds threshold
            raise "cellpose segmentation took too long. readjust pararmeters"
        return mask, table