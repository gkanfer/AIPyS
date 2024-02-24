import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
print(pm.__version__)
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
RANDOM_SEED = 8927
np.random.seed(RANDOM_SEED)
import pandas
import string
import cv2
import pdb
import random
import skimage
import seaborn as sns
import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from IPython.display import clear_output
from matplotlib.backends.backend_pdf import PdfPages

class TrainBayesImageSplit:
    '''
    The function `TrainBayesImageSplit` requires a labeled table as input, which includes the columns: `name`, `ratio`, `maskArea`, `label`, `signal_norm`, and `sd_norm`. 
    It outputs a Bayesian model. The input data is created using the `GranularityDataGen.py` function and manually labeled through a labeling dashboard application. The final output of the function is a table detailing the performance of the model.
    Parameters
    ------------
    dataFileName : string
        csv file name e.g. 'imageseq_data.csv'
    dataPath : string
        path to csv file
    '''
    def __init__(self,dataFileName,dataPath,outPath,imW = 10, imH = 5,thold = 0.5):
        self.dataFileName = dataFileName
        self.dataPath = dataPath
        self.outPath = outPath
        self.imW = imW
        self.imH = imH
        self.thold = thold
        self.df = self.loadCSV()
        

    def loadCSV(self):
        return pd.read_csv(os.path.join(self.dataPath,self.dataFileName))
    
    def matchSize(self):
        df_label = self.df
        OneNum = len(df_label.loc[df_label["label"]==1])
        ind_pheno_list = df_label.loc[df_label["label"]==1,:].index.to_list()
        ind_WT_list = df_label.loc[df_label["label"]==0,:].index.to_list()
        random.shuffle(ind_WT_list)
        ind_WT_list_sel= [ind_WT_list[i] for i in range(OneNum)]
        df_label = df_label.iloc[ind_WT_list_sel+ind_pheno_list,:]
        return df_label
        
    
    def pairDistribution(self):
        df_label = self.matchSize()
        colSel = ['name','ratio','maskArea','signal_norm','sd_norm','label']
        df_label_pair = df_label.loc[:,colSel]
        label_map = {0: "WT", 1: "Pheno"}
        df_label_pair['label'] = df_label_pair['label'].map(label_map)
        palette1 = sns.color_palette("colorblind",4)
        custom_palette = {"WT": palette1[2], "Pheno": palette1[3]}
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['font.family'] = ['serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        g = sns.pairplot(df_label_pair, hue="label", kind="kde", palette=custom_palette)
        g.fig.set_size_inches(self.imW, self.imH)  # Set the figure size (width, height) in inches
        with PdfPages(os.path.join(self.outPath,'pairplot_output.pdf')) as pdf:
            pdf.savefig(g.fig)  # Save the figure `g.fig` into the PDF file
            plt.close(g.fig) 
    
    def classify(self,Int,trace):
        '''
        Parameters
        ------------ 
        Int array of intensities
        thold float
        trace pymc object
        '''
        mu = trace.posterior['a'].mean(dim=("chain", "draw")).values + trace.posterior['b'].mean(dim=("chain", "draw")).values * Int
        prob = 1 / (1 + np.exp(-mu))
        return prob, prob > self.thold
    
    
    
    def bayesModelTraining(self):
        df_label = self.matchSize()
        rate = df_label['ratio'].values
        y_0 = df_label['label'].values
        with pm.Model() as model_logistic_basic:
            a = pm.Normal('a', 0, 2)
            b = pm.Normal('b', 0, 2)
            mu = a + pm.math.dot(rate, b)
            theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
            bd = pm.Deterministic('bd', -a / b)
            yl = pm.Bernoulli('yl', theta, observed=y_0)
            trace = pm.sample(4000, tune=4000, target_accept=0.99, random_seed=RANDOM_SEED)
        prob, prediction = self.classify(Int = rate, trace = trace)
        y_true = y_0
        y_pred = np.where(prediction == True, 1, 0)
        performance = precision_recall_fscore_support(y_true, y_pred, average='macro')
        with PdfPages(os.path.join(self.outPath,'BayesReport.pdf')) as pdf:
            plt.figure(figsize=(3, 3))
            plt.title('Trace Plot')
            az.plot_trace(trace, figsize=(12, 6), compact=True)
            az.plot_trace(trace, figsize=(12, 6), compact=True)
            pdf.savefig()
            plt.close()

            
            
            idx = np.argsort(rate)
            theta = trace.posterior['theta'].mean(dim=("chain", "draw")).values
            plt.figure(figsize=(3, 3))
            plt.title('Boundary plot')
            plt.plot(rate[idx], theta[idx], color='b', lw=3)
            plt.axvline(trace.posterior['bd'].mean(), ymax=1, color='r')
            bd_hdi = pm.hdi(trace.posterior['bd'])
            bd_low = pm.hdi(trace.posterior['bd']).sel(hdi='lower')['bd'].values
            bd_high = pm.hdi(trace.posterior['bd']).sel(hdi='higher')['bd'].values
            plt.fill_betweenx([0, 1], bd_low, bd_high, color='r')
            plt.plot(rate, y_0, 'o', color='k')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(3, 3))
            plt.title('Performance')
            confusion_matrix = metrics.confusion_matrix(np.array(df_label['label'].values, dtype=int),np.where(prediction, 1, 0))
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
            cm_display.plot()
            plt.text(0, 0.5, "Precision :{}".format(np.round(performance[0],2)), fontsize=10, transform=plt.gcf().transFigure)
            plt.text(0, 0.4, "Recall :{}".format(np.round(performance[1],2)), fontsize=10, transform=plt.gcf().transFigure)
            plt.text(0, 0.3, "F1 score :{}".format(np.round(performance[2],2)), fontsize=10, transform=plt.gcf().transFigure)
            plt.text(0, 0.2, "a :{}".format(np.round(trace.posterior['a'].mean().values, 2)), fontsize=10,transform=plt.gcf().transFigure)
            plt.text(0, 0.1, "b :{}".format(np.round(trace.posterior['b'].mean().values, 2)), fontsize=10,transform=plt.gcf().transFigure)
            pdf.savefig()
            plt.close()
            
            
        
        
        




    
    