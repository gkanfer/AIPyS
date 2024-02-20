#%%
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash import State
import plotly.express as px
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
import pandas as pd
import plotly.figure_factory as ff

dataPath = r'D:\Gil\images\pex_project\02142024\inc_origImage\imageSequence\data'
IMAGE_FOLDER = r'D:\Gil\images\pex_project\02142024\inc_origImage\imageSequence\images'
df = pd.read_csv(os.path.join(dataPath,'imageseq_data.csv'))
#float_cols = df.select_dtypes(include=['float']).columns
#df[float_cols] = df[float_cols].apply(lambda x: np.round(x, 3))
np.random.seed(42423)
#df = df.sample(frac=1).reset_index(drop=True)
#df['name'] = df['name'].apply(lambda x: f"{x}.png")
first_image_name = df.iloc[0]['name']
first_image_path = f"{IMAGE_FOLDER}/{first_image_name}"
savedContainer = {'name':[], 'ratio':[], 'intensity':[], 'sd':[], 'maskArea':[], 'label':[], 'signal_norm':[], 'sd_norm':[]}
image_names = [re.sub("_.*","",name) for name in set(df.name.tolist())]
# lookuptableFunction:
def lookUp(hover,columnName):
    name = df.loc[df[columnName]==hover,columnName]
    return name    

def openImageToPX(imageName):
  im_pil = Image.open(imageName)
  img = px.imshow(im_pil, binary_string=True, binary_backend="png", width=500, height=500,binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels = False)
  return img

# display distplot
hist_data = [df.loc[df['label']==1,'ratio'].values, df.loc[df['label']==0,'ratio'].values]
image_name_txt = [df.loc[df['label']==1,'name'].values, df.loc[df['label']==0,'name'].values]
group_labels = ['phno', 'wt']
fig = ff.create_distplot(hist_data, group_labels, bin_size=len(hist_data)/10,rug_text = image_name_txt)


# app.layout = html.Div([
#     html.Div([
#         dbc.Alert("Select images from rug plot:", color="primary"),
#         dcc.Dropdown(["ratio","intensity","sd"],placeholder='Plot type', id='select-col',value = "ratio"),
#         dcc.Graph(id="graph-dis-dcc", figure=fig, clear_on_unhover=True),
#     ],style={'width': '49%', 'display': 'inline-block'}),
#      html.Div([
#          dcc.Loading(html.Div(id='img-container'),type="circle", style={'height': '100%', 'width': '100%'})
#         ], style={'width': '49%', 'display': 'inline-block'}),
#     ])



app = dash.Dash(__name__,prevent_initial_callbacks=True,suppress_callback_exceptions=True)
application = app.server
app.layout = html.Div([
    html.Div([
        dbc.Alert("Select images from rug plot:", color="primary"),
        dcc.Dropdown(["ratio","intensity","sd","maskArea","signal_norm","sd_norm"],placeholder='Plot type', id='select-col',value = "ratio"),
        dcc.Graph(id="graph-dis-dcc", figure=fig, clear_on_unhover=True),
    ],style={'flex': '1', 'display': 'flex', 'flex-direction': 'column'}),
     html.Div([
         dcc.Loading(html.Div(id='img-container'),type="circle", style={'height': '100%', 'width': '100%'})
        ], style={'flex': '1'}),
    ],style={'display': 'flex', 'flex-direction': 'row'})

@app.callback(
    Output("graph-dis-dcc", "figure"),
    Input("select-col", "value"))
def update_plot(col):
    hist_data = [df.loc[df['label']==1,col].values, df.loc[df['label']==0,col].values]
    image_name_txt = [df.loc[df['label']==1,'name'].values, df.loc[df['label']==0,'name'].values]
    group_labels = ['phno', 'wt']
    fig = ff.create_distplot(hist_data, group_labels, bin_size=len(hist_data)/10,rug_text = image_name_txt)
    return fig

@app.callback(
    Output("img-container", "children"),
    Input("graph-dis-dcc", "hoverData"))
def display_hover(hoverData):
    if hoverData is None:
        return dash.no_update
    if 'text' in [x for x in hoverData["points"][0]]:
        img_name = hoverData["points"][0]["text"]
        if img_name is None:
            return dash.no_update
        img = openImageToPX(os.path.join(IMAGE_FOLDER,img_name))
        return [dcc.Graph(id="disp3",figure=img)]
    return dash.no_update
if __name__ == "__main__":
    app.run(debug=True)

