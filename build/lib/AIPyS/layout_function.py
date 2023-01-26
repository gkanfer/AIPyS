import xml.etree.ElementTree as xml
import tifffile as tfi
import skimage.measure as sme
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
# from PIL import fromarray
from numpy import asarray
from skimage import data, io
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage.morphology import binary_opening
from skimage.morphology import disk, remove_small_objects
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank, gaussian, sobel
from skimage.util import img_as_ubyte
from skimage import data, util
from skimage.measure import regionprops_table
from skimage.measure import perimeter
from skimage import measure
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import peak_local_max
import os
import glob
import pandas as pd
from pandas import DataFrame
from scipy.ndimage.morphology import binary_fill_holes
from skimage.viewer import ImageViewer
from skimage import img_as_float
import time
import base64
from datetime import datetime

os.chdir('/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash/app_uploaded_files/')
image = 'dmsot0273_0003.tif'
image = 'Composite.tif10.tif'

pixels = tfi.imread(image)
pixels_float = pixels.astype('float64')
pixels_float = pixels_float / 65535.000
ch = pixels_float[0, :, :]
ch2 = pixels_float[1, :, :]

import time


start =  time.time()

ll=list()
ther_list = seq(start=0.00001, end = 0.9, by=0.1, length_out=None)
len(ther_list)

start =  time.time()



plt.hist(np.ravel(ch), bins='auto')

threshold_otsu(ch)
np.median(np.ravel(ch))-threshold_otsu(ch)

nmask = threshold_local(ch, 13, "mean", np.median(np.ravel(ch))/10)
nmask2 = ch > nmask
nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
nmask4 = binary_fill_holes(nmask3)
plt.imshow(nmask4)
label_objects = sm.label(nmask4, background=0)
plt.imshow(label_objects)
info_table = pd.DataFrame(
    measure.regionprops_table(
        label_objects,
        intensity_image=ch,
        properties=['area', 'label','coords'],
    )).set_index('label')
#info_table.hist(column='area', bins='auto')
ll.append(len(info_table))
end=time.time()
print(end - start)

np.