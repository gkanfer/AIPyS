import os
os.chdir("D:\Subhra\AIPyS\AIPyS")
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
from AIPyS_old import AIPS_file_display as afd
from AIPyS_old import AIPS_cellpose as AC
from AIPyS_old.AIPS_cellpose import granularityMesure_cellpose

import glob
from AIPyS_old.Baysian_training import bayesModelTraining
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
print(pm.__version__)
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


dir1 = r'D:\Subhra\AIPyS\AIPyS\data\40X\WT\csv'
dir2 = r'D:\Subhra\AIPyS\AIPyS\data\40X\PEX3\csv'

files = glob.glob(pathname=dir1+"\*.csv") + glob.glob(pathname=dir2+"\*.csv")
dfMergeFinel, dfMergeFinelFitelrd, rate, y_0, trace = bayesModelTraining(files = files,kernelSize = 6,pathOut = None, reportFile = None)

