import base64
import datetime
import io
import plotly.graph_objs as go

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd


def parse_contents(contents, filename):
    content_string = contents[0].split('data:text/csv;base64,')[1]
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename[0]:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])