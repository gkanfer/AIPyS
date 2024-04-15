import argparse
import random
import glob
from pathlib import Path
import os
from AIPyS.segmentation.cellpose.plotObjectAreas import plotObjectAreas
from AIPyS.supportFunctions.unpack_h5 import parametersInspec
from web_app.AreasViz.app import area_viz

def main():
    parameters = parametersInspec(option,user_parameters_path)
    if parameters['channels']=='greyscale':
        channels = [0,0]
    else:
        print('channels are missing')
    pattern = os.path.join(parameters['trainingDataPath'], '**', '*.tif')
    # List all the .tif files
    files = glob.glob(pattern, recursive=True)
    random.shuffle(files)
    plotObjectAreas(kernelGran = parameters['kernelGran'],w = parameters['w'],h = parameters['h'],
                        extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],diameter =  parameters['diameter'],
                        model_type = parameters['model_type'], channels = channels,
                        Image_name = files[:parameters['imagesN']], outPath = parameters['outPath'],trainingDataPath = parameters['trainingDataPath'])
    print("Web application initiate")
    dataPath = parameters['dataPath']
    imagePath = parameters['imagePath']
    app = area_viz(imagePath,dataPath)