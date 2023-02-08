import sys
sys.path.append(r'F:\Gil\AIPS_platforms\AIPyS')
from AIPyS import AIPS_module as ai
from AIPyS import AIPS_functions as af
from AIPyS import AIPS_file_display as afd

import tifffile as tfi
import numpy as np
from PIL import Image
import plotly.express as px
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
from skimage import io, filters, measure, color, img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
from skimage.exposure import rescale_intensity
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from AIPyS.temp_function import AIPS_module_temp as aitemp



input_str ='catGFP.tif'
UPLOAD_DIRECTORY = r'F:\Gil\AIPS_platforms\AIPyS\data'
AIPS_object = ai.Segmentation(Image_name=input_str, path=UPLOAD_DIRECTORY, ch_=1, rmv_object_nuc=0.12,block_size=59, offset=-4, clean = 3)
nuc_s = AIPS_object.seedSegmentation()