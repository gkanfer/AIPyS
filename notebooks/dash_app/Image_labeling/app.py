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

df_Orig = pd.read_csv(os.path.join(dataPath,'imageOrig_data.csv'))

def reMeanSD(data,imageName):
    mean = data.loc[data['name'].str.contains(imageName),'intensity'].values
    sd = data.loc[data['name'].str.contains(imageName),'sd'].values
    return mean,sd
    
def normIntenSd(df,dfOrig,sel_name):
    curr_name = re.sub('_.*','',sel_name)
    mean,sd = reMeanSD(dfOrig,curr_name)
    mean_list_curr = df.loc[df['name'].str.contains(curr_name),'intensity'].values
    sd_list_curr = df.loc[df['name'].str.contains(curr_name),'sd'].values
    df.loc[df['name'].str.contains(curr_name),'signal_norm'] = mean_list_curr/mean
    df.loc[df['name'].str.contains(curr_name),'sd_norm'] = sd_list_curr/sd
    
    
df = pd.read_csv(os.path.join(dataPath,'imageseq_data.csv'))
float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].apply(lambda x: np.round(x, 4))
image_names = [re.sub("_.*","",name) for name in set(df.name.tolist())]
df['signal_norm'] = 0
df['sd_norm'] = 0
# normalize data frame- traverse by unique image name and add normalize intensity and sd to df 
for imag_name_curr in image_names:
    normIntenSd(df,df_Orig,imag_name_curr)

np.random.seed(42423)
df = df.sample(frac=1).reset_index(drop=True)
#df['name'] = df['name'].apply(lambda x: f"{x}.png")
first_image_name = df.iloc[0]['name']
first_image_path = f"{IMAGE_FOLDER}/{first_image_name}"
savedContainer = {'name':[], 'ratio':[], 'intensity':[], 'sd':[], 'maskArea':[], 'label':[]}
image_names = [re.sub("_.*","",name) for name in set(df.name.tolist())]

def openImageToPX(imageName):
  im_pil = Image.open(imageName)
  img = px.imshow(im_pil,binary_string=True, binary_backend="png", width=650, height=650,binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels = False)
  return img

app = dash.Dash(__name__,prevent_initial_callbacks=True,suppress_callback_exceptions=True)
application = app.server
app.layout = html.Div([
       html.Div([
            dash_table.DataTable(id='table',
            columns=[{"name": i, "id": i,}
                    for i in df.columns if i in ['name', 'ratio', 'label']],
            data=df.to_dict('records'),
            row_selectable="single",
            editable = True,
            page_size=10,
            page_current = 0,
            page_action='native',
            row_deletable=True,
            cell_selectable=True),
            dbc.Button('Save data', id='save-button', color="danger", n_clicks=0, active=True),
            dcc.Loading(html.Div(id='load-csv-file'), type="circle", style={'height': '100%', 'width': '100%'}),
        ],style={'flex': '1', 'display': 'flex', 'flex-direction': 'column'}),
        html.Div([
          html.Div(id='img-container')
        ], style={'flex': '1'}),
        html.Div(id='temp-contain'),
],style={'display': 'flex', 'flex-direction': 'row'})

@app.callback(
  Output('img-container', 'children'),
  [Input('table', 'active_cell'),
  State('table', 'page_current'),
  State('table', 'page_size')],
  prevent_initial_call=True,)
def display_image(active_cell,page_current,page_size):
  '''
  active_cell - dictionery {'row': 0, 'column': 0, 'column_id': 'name'}
  rows - list of dict per row:  [{'Unnamed: 0': 294, 'name': 'QOnvDbK_24.png', 'ratio': 0.032,
  'intensity': 783.206, 'sd': 611.006, 'maskArea': 0.995, 'label': 0},....]
  '''
  if (active_cell is None) and (page_current == 0):
    img_name = df.loc[0,'name']
    img = openImageToPX(os.path.join(IMAGE_FOLDER,img_name))
    return [dcc.Graph(id="disp3",figure=img)]
  elif (active_cell is None) and (page_current > 0):
    return ["select image name"]
  elif (active_cell['column_id'] == 'name') and (page_current == 0):
    img_name = df.loc[active_cell['row'],active_cell['column_id']]
    img = openImageToPX(os.path.join(IMAGE_FOLDER,img_name))
    return [dcc.Graph(id="disp2",figure=img)]
  elif  (active_cell['column_id'] == 'name') and (page_current > 0):
    df_new = df.iloc[page_current*page_size:(page_current+1)*page_size].reset_index()
    img_name = df_new.loc[active_cell['row'],active_cell['column_id']]  # No need to append ".png"
    img = openImageToPX(os.path.join(IMAGE_FOLDER,img_name))
    return [dcc.Graph(id="disp1",figure=img)]
  raise PreventUpdate

@app.callback(
  Output('load-csv-file', 'children'),
  [Input('save-button', 'n_clicks'),
   State('table', 'data')],
  prevent_initial_call=True,
)
def save_data(n,curr_table):
  # Handler for data update; includes save logic as needed
  if n is None:
    return dash.no_update
  else:
    df_curr = pd.DataFrame(curr_table).reset_index()
    df_curr.to_csv(os.path.join(dataPath,'imageseq_data.csv'),index=False)
    return "data is saved"

if __name__ == '__main__':
  app.run(debug=True)




# %%
