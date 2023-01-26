'''
function description:
Binary byes' classification

Input table should be two classified tables e.g.: normal vs phenotype
Paramters:
1) prior flags: optional
2)
'''

import pandas as pd
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

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

from utils import AIPS_module as ai
from utils import AIPS_functions as af
from utils import AIPS_granularity as ag
from utils.AIPS_cellpose import granularityMesure_cellpose



def bayesModelTraining(files,kernelSize,pathOut, reportFile):
    dfMergeFinel = ag.MERGE().mergeTable(tableInput_name_list=files)
    dfMergeFinelFitelrd = ag.MERGE().calcDecay(dfMergeFinel, kernelSize)

    # plot information before training
    def generate_plots():
        def line():
            dfline = pd.DataFrame(
                {"kernel": dfMergeFinel.kernel.values, "Signal intensity (ratio)": dfMergeFinel.intensity.values,
                 "class": dfMergeFinel.classLabel.values})
            fig, ax = plt.subplots()
            sns.lineplot(data=dfline, x="kernel", y="Signal intensity (ratio)", hue="class").set(
                title='Granularity spectrum plot')
            return ax
        def plotBox():
            classLabel = dfMergeFinelFitelrd.classLabel.values.tolist()
            intensity = dfMergeFinelFitelrd.intensity.values.tolist()
            df = pd.DataFrame({"classLabel": classLabel, "intensity": intensity})
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="classLabel", y="intensity").set(title='Cell area distribution')
            return ax
        plot1 = plotBox()
        plot2 = line()
        return (plot1, plot2)

    def plots2pdf(plots, fname):
        with PdfPages(fname) as pp:
            for plot in plots:
                pp.savefig(plot.figure)
    plots2pdf(generate_plots(), os.path.join(pathOut, 'preTrainingPlots.pdf'))

    #baysian training
    rate = dfMergeFinelFitelrd.intensity.values
    y_0 = dfMergeFinelFitelrd.classLabel.values
    with pm.Model() as model_logistic_basic:
        a = pm.Normal('a', 0, 10)
        b = pm.Normal('b', 0, 10)
        mu = a + pm.math.dot(rate, b)
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
        bd = pm.Deterministic('bd', -a / b)
        yl = pm.Bernoulli('yl', theta, observed=y_0)
        trace = pm.sample(4000, tune=4000, target_accept=0.99, random_seed=RANDOM_SEED, cores=1)
    # performance table
    def classify(n, thold):
        mu = trace['a'].mean() + trace['b'].mean() * n
        prob = 1 / (1 + np.exp(-mu))
        return prob, prob > thold
    rate = dfMergeFinelFitelrd.intensity.values
    td = 0.5
    prob, prediction = classify(rate, td)
    y_true = y_0
    y_pred = np.where(prediction == True, 1, 0)
    performance = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # Plot
    with PdfPages(os.path.join(pathOut, reportFile + '.pdf')) as pdf:
        plt.figure(figsize=(3, 3))
        plt.title('Trace Plot')
        az.plot_trace(trace, figsize=(12, 6), compact=True)
        pdf.savefig()
        plt.close()

        idx = np.argsort(rate)
        theta = trace['theta'].mean(0)
        plt.figure(figsize=(3, 3))
        plt.title('Boundary plot')
        plt.plot(rate[idx], theta[idx], color='b', lw=3)
        plt.axvline(trace['bd'].mean(), ymax=1, color='r')
        bd_hdi = pm.hdi(trace['bd'])
        plt.fill_betweenx([0, 1], bd_hdi[0], bd_hdi[1], color='r')
        plt.plot(rate, y_0, 'o', color='k')
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(3, 3))
        plt.title('Performance')
        confusion_matrix = metrics.confusion_matrix(np.array(dfMergeFinelFitelrd.classLabel.values, dtype=int),np.where(prediction, 1, 0))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
        cm_display.plot()
        plt.text(0.07, 0.5, "Precision :{}".format(np.round(performance[0],2)), fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.06, 0.4, "Recall :{}".format(np.round(performance[1],2)), fontsize=10, transform=plt.gcf().transFigure)
        plt.text(0.05, 0.4, "F1 score :{}".format(np.round(performance[2],2)), fontsize=10, transform=plt.gcf().transFigure)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(3, 3))
        plt.title('Variables')
        plt.text(0.07, 0.5, "a :{}".format(np.round(trace['a'].mean(), 2)), fontsize=10,transform=plt.gcf().transFigure)
        plt.text(0.06, 0.4, "b :{}".format(np.round(trace['b'].mean(), 2)), fontsize=10,transform=plt.gcf().transFigure)
        pdf.savefig()
        plt.close()




