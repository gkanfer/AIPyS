#%%
import dash
from dash import html
from dash import dash_table, dcc
from dash.dependencies import Input, Output
#import dash_core_components as dcc
from dash import State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
import re
import glob
import random
from sklearn import preprocessing
from tqdm import tqdm
import plotly.express as px

dataPath = r'D:\Gil\images\pex_project\02142024\inc_origImage\imageSequence\data'
IMAGE_FOLDER = r'D:\Gil\images\pex_project\02142024\inc_origImage\imageSequence\images'
# normlise intensity and sd and mask area by origImage.
#%%
df_Orig = pd.read_csv(os.path.join(dataPath,'imageOrig_data.csv'))
#%%
def reMeanSD(data,imageName):
    mean = data.loc[data['name'].str.contains(imageName),'intensity'].values
    sd = data.loc[data['name'].str.contains(imageName),'sd'].values
    return mean,sd
    
def normIntenSd(df,dfOrig):
    curr_name = re.sub('_.*','',df.loc[0,'name'])
    mean,sd = reMeanSD(dfOrig,curr_name)
    mean_list_curr = df.loc[df['name'].str.contains(curr_name),'intensity'].values
    sd_list_curr = df.loc[df['name'].str.contains(curr_name),'sd'].values
    df.loc[df['name'].str.contains(curr_name),'intensity'] = mean_list_curr/mean
    df.loc[df['name'].str.contains(curr_name),'intensity'] = sd_list_curr/sd
    
    
df = pd.read_csv(os.path.join(dataPath,'imageseq_data.csv'))
float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].apply(lambda x: np.round(x, 3))
normIntenSd(df,df_Orig)
np.random.seed(42423)
df = df.sample(frac=1).reset_index(drop=True)
# %%
