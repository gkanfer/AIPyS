import numpy as np
import pandas as pd
from skimage import io, data
from PIL import Image
import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate
import json
import dash_canvas
from dash_canvas.utils import (image_string_to_PILImage, array_to_data_url)


def parse_jsonstring_line(string):
    """
    Return geometry of line objects.

    Parameters
    ----------

    data : str
        JSON string of data

    """
    try:
        data = json.loads(string)
    except:
        return None
    scale = 1
    props = []
    for obj in data['objects']:
        if obj['type'] == 'image':
            scale = obj['scaleX']
        elif obj['type'] == 'line':
            length = np.sqrt(obj['width']**2 + obj['height']**2)
            scale_factor = obj['scaleX'] / scale
            props.append([scale_factor * length,
                          scale_factor * obj['width'],
                          scale_factor * obj['height'],
                          scale_factor * obj['left'],
                          scale_factor * obj['top']])
    return (np.array(props)).astype(int)


def measure_length(filename):
    '''
    file name is string of the path for example file
    '''
    try:
        img = io.imread(filename)
        input_gs_image = img
        input_gs_image = input_gs_image*2
        input_gs_image = (input_gs_image / input_gs_image.max()) * 255
        img = np.uint8(input_gs_image)
    except:
        img = data.coins()
    height, width = img.shape
    canvas_width = 500
    canvas_height = round(height * canvas_width / width)
    scale = canvas_width / width

    list_columns = ['length', 'width', 'height']
    columns = [{"name": i, "id": i} for i in list_columns]

    app = dash.Dash(__name__)
    server = app.server
    app.config.suppress_callback_exceptions = True


    app.layout = html.Div([
        html.Div([
            dash_canvas.DashCanvas(
                id='canvas-line',
                width=canvas_width,
                height=canvas_height,
                scale=scale,
                lineWidth=2,
                lineColor='red',
                tool="line",
                hide_buttons=['pencil'],
                image_content=array_to_data_url(img),
                goButtonTitle='Measure',
                ),
        ], className="seven columns"),
        html.Div([
        html.Img(src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', width='30px'),
        html.A(
            id='gh-link',
            children=[
            'View on GitHub'
            ],
            href="http://github.com/plotly/canvas-portal/"
                              "blob/master/apps/measure-length/app.py",
            style={'color': 'black',
                'border':'solid 1px black',
                'float':'left'}
                ),

            html.H3('Draw lines and measure lengths'),
            html.H3(children='How to use this app', id='measure-subtitle'),
            html.Img(id='measure-help',
                     src='assets/measure.gif',
                     width='100%'),
            html.H4(children="Objects properties"),
            dash_table.DataTable(
                id='table-line',
                columns=columns,
                editable=True,
                ),
        ], className="four columns"),
        ])



    @app.callback(Output('canvas-line', 'tool'),
                    [Input('canvas-line', 'image_content')])
    def modify_tool(string):
        return "line"


    @app.callback(Output('measure-subtitle', 'children'),
                    [Input('canvas-line', 'json_data')])
    def reduce_help(json_data):
        if json_data:
            return ''
        else:
            return dash.no_update 


    @app.callback(Output('measure-help', 'width'),
                    [Input('canvas-line', 'json_data')])
    def reduce_help(json_data):
        if json_data:
            return '0%'
        else:
            return dash.no_update 


    @app.callback(Output('table-line', 'data'),
                    [Input('canvas-line', 'json_data')])
    def show_table(string):
        props = parse_jsonstring_line(string)
        df = pd.DataFrame(props[:, :3], columns=list_columns)
        return df.to_dict("records")
    return app


