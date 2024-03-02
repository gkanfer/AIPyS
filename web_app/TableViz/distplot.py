#%%
import io
import base64

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.express as px

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
from PIL import Image
import plotly.figure_factory as ff

dataPath = r'D:\Gil\images\pex_project\02142024\single_images\imageSequence\data'
IMAGE_FOLDER = r'D:\Gil\images\pex_project\02142024\single_images\imageSequence\images'
df = pd.read_csv(os.path.join(dataPath,'imageseq_data.csv'))
# float_cols = df.select_dtypes(include=['float']).columns
# df[float_cols] = df[float_cols].apply(lambda x: np.round(x, 3))
# np.random.seed(42423)
# df = df.sample(frac=1).reset_index(drop=True)
# #df['name'] = df['name'].apply(lambda x: f"{x}.png")
# first_image_name = df.iloc[0]['name']
# first_image_path = f"{IMAGE_FOLDER}/{first_image_name}"
# savedContainer = {'name':[], 'ratio':[], 'intensity':[], 'sd':[], 'maskArea':[], 'label':[]}
# image_names = [re.sub("_.*","",name) for name in set(df.name.tolist())]
#%%
hist_data = [df.loc[df['label']==1,'ratio'].values, df.loc[df['label']==0,'ratio'].values]
image_name_txt = [df.loc[df['label']==1,'name'].values, df.loc[df['label']==0,'name'].values]
group_labels = ['phno', 'wt']
fig = ff.create_distplot(hist_data, group_labels, bin_size=len(hist_data)/10,rug_text = image_name_txt)
fig.show()


#intensity
#%% 
hist_data = [df.loc[df['label']==1,'intensity'].values, df.loc[df['label']==0,'intensity'].values]
image_name_txt = [df.loc[df['label']==1,'name'].values, df.loc[df['label']==0,'name'].values]
group_labels = ['phno', 'wt']

fig = ff.create_distplot(hist_data, group_labels, bin_size=len(hist_data)/2,rug_text = image_name_txt)
fig.show()

#sd
#%% 
hist_data = [df.loc[df['label']==1,'sd'].values, df.loc[df['label']==0,'sd'].values]
image_name_txt = [df.loc[df['label']==1,'name'].values, df.loc[df['label']==0,'name'].values]
group_labels = ['phno', 'wt']

fig = ff.create_distplot(hist_data, group_labels, bin_size=len(hist_data)/2,rug_text = image_name_txt)
fig.show()


