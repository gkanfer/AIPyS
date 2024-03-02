import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
import seaborn as sns
#sns.set()
import arviz as az
import pymc3 as pm
print(pm.__version__)
import theano.tensor as tt
import patsy

import os
import re
import glob
import random
# import plotnine
from sklearn import preprocessing
from tqdm import tqdm

import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from skimage import measure, restoration,morphology
from skimage import io, filters, measure, color, img_as_ubyte
from skimage.draw import disk
from skimage import measure, restoration,morphology

RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
from scipy.special import expit as logistic

class BayesianModels():
    def __init__(self, df):
        self.df = df

    def getValues(self,geneSelect):
        valueInput = self.df.loc[self.df.Gene.str.contains(geneSelect),'realFoldChange'].values
        return valueInput

    def PossionRegression(self):
        RANDOM_SEED = 8927
        hitMap = {"Gene":[],"Target_stdLog2Raw":[],"non_stdLog2Raw":[],"Target_Posterior":[] ,"non_Posterior":[] }
        # standertized log2 fold change
        nonStdLog =  self.getValues(geneSelect = 'non')
        Genes = self.df.Gene.unique().tolist()
        Genes.remove('non')
        for gene in tqdm(Genes):
            # values of selected gene
            CurrValueStdLog = self.getValues(geneSelect =gene)
            GeneID = np.hstack([np.repeat(0, len(nonStdLog)), np.repeat(1, len(CurrValueStdLog))])
            y_all = np.hstack([nonStdLog,CurrValueStdLog])
            with pm.Model() as m11_12:
                a = pm.Normal("a", 0.0, 1)
                b = pm.Normal("b", 0.0, 1)
                lam = pm.math.exp(a + b * GeneID)
                y = pm.Poisson("y", lam, observed=y_all)
                trace = pm.sample(1000, tune=2000, random_seed=RANDOM_SEED)
            x = 1
            non = np.exp(trace["posterior"]["a"])
            target = np.exp(trace["posterior"]["a"] + trace["posterior"]["b"])
            hitMap['Gene'].append(gene)
            hitMap['Target_stdLog2Raw'].append(np.mean(CurrValueStdLog))
            hitMap['non_stdLog2Raw'].append(np.mean(nonStdLog))
            hitMap['Target_Posterior'].append(target)
            hitMap['non_Posterior'].append(non)
