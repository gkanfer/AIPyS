import sys
sys.path.append(r'F:\Gil\AIPS_platforms\AIPyS')
import glob
from AIPyS.Baysian_training import bayesModelTraining
import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc3 as pm
print(pm.__version__)
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)


pathIn =   r'F:\Gil\AIPS_platforms\AIPyS\data'
files = glob.glob(pathname=pathIn+"\*.csv")
rate, y_0, trace = bayesModelTraining(files = files,kernelSize = 5,pathOut = None, reportFile = None, savemode = False)