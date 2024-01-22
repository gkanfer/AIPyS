import os
os.chdir("D:\Gil\AIPyS-main\AIPyS")
import seaborn as sns
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from skimage import io
import sys
from AIPyS import AIPS_file_display as afd
from AIPyS import AIPS_cellpose as AC
from AIPyS.AIPS_cellpose import granularityMesure_cellpose

path_WT =  r'D:\Gil\40X\PEX3'
path_KO = r'D:\Gil\40X\WT'

WTFileNames = os.listdir(path_WT)
KOfilenames = os.listdir(path_KO)

WTcompsiteImage, WTtable, WTdf, WTgTable = granularityMesure_cellpose(file = WTFileNames[0],path = path_WT, classLabel = 0,outPath = None, clean = 2000, outputTableName = None,saveMode=False,intensity = 1,start_kernel=2, end_karnel=20, kernel_size=10)
kOcompsiteImage, kOtable, kOdf, kOgTable = granularityMesure_cellpose(file = KOfilenames[0],path = path_KO, classLabel = 1,outPath = None, clean = 2000, outputTableName = None,saveMode=False,intensity = 1,start_kernel=2, end_karnel=20, kernel_size=10)

# save wt examples has cvs
import tqdm
import random

path_WT =  r'D:\Gil\40X\WT'
path_KO = r'D:\Gil\40X\PEX3'
WTFileNames = [x for x in os.listdir(path_WT) if '.tif' in x]
random.shuffle(WTFileNames)
KOfilenames =[x for x in os.listdir(path_KO) if '.tif' in x]
random.shuffle(KOfilenames)
def saveGranularityData(filenamesList,outName,inPath,outPath,classLabel,itrNum):
    index = 0
    for i in tqdm.tqdm(range(3,50)):
        try:
            compsiteImage, table, df, gTable = granularityMesure_cellpose(file = filenamesList[i], path = inPath, classLabel = classLabel,outPath = None, clean = 2000, outputTableName = None,saveMode=False,intensity = 1,start_kernel=2, end_karnel=20, kernel_size=10)
        except:
            print("no cells detected")
            table = []
        index += len(table)
        print(f'number of cells: {index}')
        if index > 2500:
            break
        else:
            gTable.to_csv(os.path.join(outPath,outName +"_"+str(i)+".csv"))
#saveGranularityData(filenamesList = WTFileNames,outName = 'wt',inPath = path_WT, outPath = r'D:\Gil\40X\WT\csv', classLabel = 0, itrNum = 10)
saveGranularityData(filenamesList = KOfilenames,outName = 'ko',inPath = path_KO, outPath = r'D:\Gil\40X\PEX3\csv', classLabel = 1, itrNum = 10)



