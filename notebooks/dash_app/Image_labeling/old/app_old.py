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

dataPath = r'D:\Gil\images\pex_project\02142024\single_images\imageSequence\data'
IMAGE_FOLDER = r'D:\Gil\images\pex_project\02142024\single_images\imageSequence\images'
df = pd.read_csv(os.path.join(dataPath,'imageseq_data.csv'))
float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].apply(lambda x: np.round(x, 3))
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

# im_pil = Image.open(os.path.join(IMAGE_FOLDER,df.loc[0,'name']))
# fig = px.imshow(im_pil,zmin=[50, 50, 50], zmax=[100, 100, 255],binary_string=True, binary_backend="png", width=650, height=650,binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels = False)
# #zmin=[50, 50, 50], zmax=[255, 255, 255]
# fig.show()


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
        ],style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
          html.Div(id='img-container')
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div(id='temp-contain'),
])

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








# @app.callback(
#   Output('temp-contain', 'children'),
#   [Input('table', 'active_cell'),
#    Input('table', 'data'),
#    State('table', 'page_current'),
#    State('table', 'page_size')],
#   prevent_initial_call=True,
# )
# def update_data(active_cell,curr_table,page_current,page_size):
#   # Handler for data update; includes save logic as needed
#   if (active_cell is None) and (page_current == 0):
#     return [""]
#   elif (active_cell is None) and (page_current > 0):
#     return [""]
#   elif (active_cell['column_id'] == 'label') and (page_current==0):
#     df_curr = pd.DataFrame(curr_table)
#     label = df_curr.loc[active_cell['row'],'label']
#     df.loc[active_cell['row'],'label'] = label
#     df.to_csv(os.path(dataPath,'imageseq_data.csv'))
#     return [""]
#   elif (active_cell['column_id'] == 'label') and (page_current > 0):
#     df_curr = pd.DataFrame(curr_table)
#     label = df_curr.loc[page_current*page_size+active_cell['row'],'label']
#     df.loc[page_current*page_size+active_cell['row'],'label'] = label
#     df.to_csv(os.path(dataPath,'imageseq_data.csv'))
#     return [""]
  
# if __name__ == '__main__':
#   app.run(debug=True)


