import dash_daq as daq
import dash
import dash.exceptions
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import tifffile as tfi
import glob
import os
import numpy as np
from skimage.exposure import rescale_intensity, histogram
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import base64
import pandas as pd
import re
from random import randint
from io import BytesIO

# decimal 1 to 99 global variable
integer_1_99 = list(np.around(np.arange(1, 99, 10),1)-1)
integer_1_99[0] = 1
integer_1_99.append(99)

controls = dbc.Card(
        [html.Div(
                [
                    dbc.Label("Choose seed channel"),
                    html.Div([
                        html.Div(dcc.Dropdown('Cahnnel 1', id='seed-channeLselect',value = 0, className="inner-box-item")
                                 , className="inner-box-image"),
                        html.Div(dcc.Dropdown('Cahnnel 2', id='target-channeLselect',value=1 , className="inner-box-item")
                                 , className="inner-box-image"),
                    ], className="div-image-video"),
                    dcc.RadioItems(
                        id='act_ch',
                        options=[
                            {'label': 'Channel 1', 'value': 0},
                            {'label': 'Channel 2', 'value': 1},
                        ],
                        value=0,
                        labelStyle={'display': 'inline-block'}
                    )
                ]),
            html.Div(
                [
                    dbc.Label("Intensity factor"),
                    dcc.Slider(
                        id='int-factor',
                        min=0.1,
                        max=10,
                        step=0.1,
                        marks={i:'{}'.format(i) for i in [0.1,10]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=1,
                        ),
                ]
            ),
            html.Br(),
            html.Br(),
            # redusing memory
        ], body=True,className='card')



controls_nuc = dbc.Card(
        [
            html.Div(
                [
                    dbc.Label("Auto parameters initialise"),
                    dbc.Row([
                        dbc.Col(
                            daq.BooleanSwitch(
                                id='Auto-nuc',
                                disabled=False,
                                on=True,
                            )),
                            dbc.Col([
                                daq.GraduatedBar(
                                        id='graduated-bar',
                                        label="Search more",
                                        value=1,
                                        min=1,
                                        max=10
                                        ),
                                dcc.Slider(
                                    id='graduated-bar-slider',
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=1
                                ),
                            ]),

                            dbc.Col([
                                daq.GraduatedBar(
                                    id='graduated-bar-nuc-zoom',
                                    label="Zoom in filter",
                                    value=5,
                                    min=1,
                                    max=100
                                ),
                                dcc.Slider(
                                    id='graduated-bar-slider-nuc-zoom',
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=5
                                ),
                            ]),
                        ]),
                    ]),
            html.Div(
                [
                    dbc.Label("Detect nuclei edges:"),
                    dcc.Slider(
                        id='offset',
                        min=1,
                        max=255,
                        step=1,
                        marks={i: '{}'.format(i) for i in [1, 255]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=100
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Nucleus segmentation"),
                    html.Br(),
                    dbc.Label("Local Threshold:"),
                    dcc.Slider(
                        id='block_size',
                        min=1,
                        max=201,
                        step=2,
                        marks={i: '{}'.format(i) for i in [1, 201]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=59,
                        #tooltip = { 'always_visible': True }
                        ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove small objects:"),
                    dcc.Slider(
                        id='rmv_object_nuc',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: '{}'.format(i) for i in [0.01,0.99]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=0.9
                    ),
                ]
                ),
            html.Div([
                    dbc.Button("Plus",
                           id = "plus-cleanSeed",
                            color="primary",
                            n_clicks=0),
                    html.Div(id = "cleanSeed"),
                    dbc.Button("Minus",
                               id="minus-cleanSeed",
                               color="primary",
                               n_clicks=0),
                    dcc.Store(id = "cleanCounter-seed",data = None),
            ])
         ],
        body=True,
            className='card'
    )


controls_cyto = dbc.Card(
        [
            html.Div(
                [
                    dbc.Label("Detect cytosol edges:"),
                    dcc.Slider(
                        id='offset_cyto',
                        min=-100,
                        max=256,
                        step=1,
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=100,
                        #tooltip = { 'always_visible': True }
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Local Threshold:"),
                    dcc.Slider(
                        id='block_size_cyto',
                        min=1,
                        max=201,
                        step=2,
                        marks={i: '{}'.format(i) for i in [1, 201]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=13,
                        #tooltip = { 'always_visible': True }
                        ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Detect global edges:"),
                    dcc.Slider(
                        id='global_ther',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: '{}'.format(i) for i in [0.01, 0.99]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=0.3
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove large objects:"),
                    dcc.Slider(
                        id='rmv_object_cyto',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: '{}'.format(i) for i in [0.01, 0.99]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=0.99
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Label("Remove small objects:"),
                    dcc.Slider(
                        id='rmv_object_cyto_small',
                        min=0.01,
                        max=0.99,
                        step=0.01,
                        marks={i: '{}'.format(i) for i in [0.01, 0.99]},
                        tooltip={"placement": "bottom", "always_visible": True},
                        value=0.99
                    ),
                ]
            ),
         ],
        body=True,
        className='card'
    )

upload_parm = dbc.Row([
                dbc.Col([
                     dcc.Upload(
                            id='upload-csv',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                    html.Button('Upload parameters', id='submit-parameters', n_clicks=0)])
                    ],className='card')

svm_slice_slider =  html.Div(
                    [
                    dbc.Label("Remove large objects:"),
                    dcc.Slider(
                        id='slice_slider',
                        min=1,
                        max=16,
                        step=4,
                        marks={i: i for i in [1,4,8,16]},
                        value=1
                    ),
                    ],className='card'
                    )

def generate_team_button(Name):
    return dbc.Button(children=str(Name),
                      color="primary",
                      className="mr-1",
                      id=str(Name))