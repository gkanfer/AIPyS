# Measuring Cell Diameter
# python measure_diameter.py --debug
import argparse
import random
import glob
from pathlib import Path
import os

import threading
import webbrowser
import time

from web_app.measure_length.app import measure_length
from web_app.Image_labeling.app import image_labeling
from web_app.TableViz.app import data_viz
from AIPyS.segmentation.cellpose.AIPS_cellpse_video_gen import CellPoseSeg
from AIPyS.classification.bayes.GranulaityMesure import GranulaityMesure_cp
from AIPyS.classification.bayes.GranularityDataGen import GranularityDataGen_cp
from AIPyS.classification.bayes.Baysian_training import Baysian_training
from AIPyS.classification.bayes.Baysian_deploy import BayesDeploy
from AIPyS.supportFunctions.unpack_h5 import parametersInspec
from AIPyS.CLI.set_parameters import update_auto_parameters


# location of parameters file
user_parameters_path = os.path.join(Path.home(), '.AIPyS', 'parameters.h5')

def run(option):
    if option == "measDia":
        parameters = parametersInspec(option,user_parameters_path)
        image_name = parameters["Image_name"]
        app = measure_length(image_name)
        app.run_server()
    elif option == "cp_seg_video":
        parameters = parametersInspec(option,user_parameters_path)
        files = glob.glob(parameters['data_dir'] + "\\*.tif")
        if parameters['channels']=='greyscale':
            channels = [0,0]
        else:
            print('chanlles are missing')
        CellPoseSeg(diameter =  parameters['diameter'],videoName = parameters['videoName'], model_type = parameters['model_type'], channels = channels,
                    Image_name = files[:parameters['imagesN']], outPath = parameters['outPath'])
    elif option == "cp_gran_video":
        # Placeholder for cp_gran_video option - creates a video of kernel opening
        parameters = parametersInspec(option,user_parameters_path)
        if parameters['channels']=='greyscale':
            channels = [0,0]
        else:
            print('channels are missing')
        GranulaityMesure_cp(start_kernel = parameters['start_kernel'],end_karnel = parameters['end_karnel'],kernel_size = parameters['kernel_size'],
                            extract_pixel = parameters['extract_pixel'],outputImageSize = parameters['outputImageSize'], resize_pixel = parameters['resize_pixel'],
                            videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                            Image_name = parameters['Image_name'], outPath = parameters['outPath'],diameter =  parameters['diameter']) # save video form cell segmentation
    elif option == "cp_gran_table_gen":
        parameters = parametersInspec(option,user_parameters_path)
        if parameters['channels']=='greyscale':
            channels = [0,0]
        else:
            print('channels are missing')
        pattern = os.path.join(parameters['trainingDataPath'], '**', '*.tif')
        # List all the .tif files
        files = glob.glob(pattern, recursive=True)
        random.shuffle(files)
        GranularityDataGen_cp(kernelGran = parameters['kernelGran'],w = parameters['w'],h = parameters['h'],
                            extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],diameter =  parameters['diameter'],
                            model_type = parameters['model_type'], channels = channels,
                            Image_name = files[:parameters['imagesN']], outPath = parameters['outPath'],trainingDataPath = parameters['trainingDataPath'])  
        # Placeholder for table generation - ration tables for labeling with corresponding images
    elif option == "dataLabeling":
        # Placeholder for app for labeling images
        parameters = parametersInspec(option,user_parameters_path)
        dataPath = parameters['dataPath']
        imagePath = parameters['imagePath']
        app = image_labeling(imagePath,dataPath)
        app.run_server()
        #port = 8050
        #app.run_server(port = port)
        #webbrowser.open_new(f'http://localhost:{port}')
    elif option == "data_viz":
        # Placeholder for app for data vitalization after labeling
        parameters = parametersInspec(option,user_parameters_path)
        dataPath = parameters['dataPath']
        imagePath = parameters['imagePath']
        app = data_viz(imagePath,dataPath)
        app.run_server()
    elif option == "modelBuild":
        # Placeholder for modelBuild option functionality
        parameters = parametersInspec(option,user_parameters_path)
        if parameters['channels']=='greyscale':
            channels = [0,0]
        else:
            print('channels are missing')
        model = Baysian_training(dataPath = parameters['dataPath'], imW = parameters['imW'], imH = parameters['imH'], thold = parameters['thold'],
                        extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],
                         model_type = parameters['model_type'], channels = channels,Image_name = parameters['Image_name'],
                         outPath = parameters['outPath'], diameter =  parameters['diameter'],areaSel = parameters['areaSel'],fractionData = parameters['fractionData']).bayesModelTraining()
        updateParaDict = {"intercept":model.intercept,"slope":model.slope}
        update_auto_parameters(parameter_updates = updateParaDict.intercept, user_parameters_path = user_parameters_path.slope)
        print("Granularity model parameters estimation is done")
    elif option == "deployBuild":
        # Placeholder for deployBuild option functionality
        parameters = parametersInspec(option,user_parameters_path)
        if parameters['channels']=='greyscale':
            channels = [0,0]
        else:
            print('channels are missing')
        parameters = parametersInspec(option,user_parameters_path)
        BayesDeploy(intercept = parameters['intercept'], slope = parameters['slope'], kernelGran = parameters['kernelGran'], dataPath = parameters['dataPath'], imW = parameters['imW'], imH = parameters['imH'], thold = parameters['thold'],
                    extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],areaSel = parameters['areaSel'],fractionData = parameters['fractionData'],
                     model_type = parameters['model_type'], channels = channels, Image_name = parameters['Image_name'], outPath = parameters['outPath'], diameter =  parameters['diameter'],)
    else:
        print('Invalid option. Please choose a valid option.')


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Application for measuring cell diameter.')
    # Add an argument for the option selection
    parser.add_argument('--option', type=str, # Added the long option '--option' for clarity
                        choices=['measDia', 'cp_seg_video', 'cp_gran_video', 'cp_gran_table_gen', 'dataLabeling', 'data_viz', 'modelBuild', 'deployBuild'],
                        required=True,
                        help='Select an option to run: measDia, cp_seg_video, cp_gran_video, cp_gran_table_gen, dataLabeling, data_viz, modelBuild, deployBuild')
    args = parser.parse_args()

    # Call the run function with the selected option and debug mode
    run(args.option)

