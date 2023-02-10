import tifffile as tfi
import numpy as np
from PIL import Image
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
from skimage import io
from PIL import ImageEnhance
import sys
sys.path.append(r'F:\Gil\AIPS_platforms\AIPyS')
from AIPyS import AIPS_file_display as afd
from AIPyS import AIPS_cellpose as AC
from AIPyS.AIPS_cellpose import granularityMesure_cellpose

fileNmae = ['PEX3KO.tif','WT.tif']
path = '/data/kanferg/Images/Pex_project/SIngle_cell_images_training_set/AIPyS_Images'
path = r'F:\Gil\AIPS_platforms\AIPyS\data'
WTcompsiteImage, WTtable, WTdf = granularityMesure_cellpose(file = fileNmae[1],path = path, classLabel = 0,outPath = None, clean = None, outputTableName = None,saveMode=False,intensity = 1,start_kernel=2, end_karnel=60, kernel_size=10)
kOcompsiteImage, kOtable, kOdf = granularityMesure_cellpose(file = fileNmae[0],path = path, classLabel = 1,outPath = None, clean = None, outputTableName = None,saveMode=False, intensity = 1,start_kernel=2, end_karnel=60, kernel_size=10)
# PEX3KO = io.imread(os.path.join(path,fileNmae[0]))
# WT = io.imread(os.path.join(path,fileNmae[1]))
# adjust the brightness
# PEX3KO = io.imread(os.path.join(path,fileNmae[0]))
# WT = io.imread(os.path.join(path,fileNmae[1]))
# adjust the brightness
# WT = ImageEnhance.Brightness(WTcompsiteImage)
# WT_dimmer = WT.enhance(0.75)
# PEX = ImageEnhance.Brightness(PEX3KOcompsiteImage)
# Pex_dimmer = PEX.enhance(0.65)
# fig, ax = plt.subplots(1, 2, figsize=(8, 8))
# ax[0].imshow(WT_dimmer, cmap=plt.cm.gray)
# ax[0].title.set_text('WT')
# ax[1].imshow(Pex_dimmer, cmap=plt.cm.gray)
# ax[1].title.set_text('PEX3KO')



