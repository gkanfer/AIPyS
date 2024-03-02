# Measuring Cell Diameter
# python measure_diameter.py --debug
import argparse
import sys
import h5py
from pathlib import Path
import os
from web_app.measure_length.app import measure_length
from web_app.Image_labeling.app import image_labeling
from web_app.TableViz.app import data_viz
from AIPyS.classification.bayes.GranulaityMesure import GranulaityMesure_cp
from AIPyS.classification.bayes.GranularityDataGen import GranularityDataGen_cp
from AIPyS.supportFunctions.unpack_h5 import unpack_h5
from AIPyS.classification.bayes.Baysian_training import Baysian_training
from AIPyS.classification.bayes.BayesDeploy import BayesDeploy
# location of parameters file
user_parameters_path = os.path.join(Path.home(), '.AIPyS', 'parameters.h5')

# "measDia","cp_seg","cp_gran","modelBuild","deployBuild"
def run(option,file_number,user_parameters_path):
    parameters = unpack_h5(user_parameters_path)
    if option == "measDia":
        image_name = parameters['Image_name']
        app = app_measLen(image_name)
        app.run_server()
    elif option == "cp_seg_video":
        files = parameters['Image_name'][:file_number] # should be from glob
        AIPS_cellpse_video_gen(videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                                Image_name = files, outPath = parameters['outPath']) # save video form cell segmentation
    elif option == "cp_gran_video":
        # Placeholder for cp_gran_video option - creates a video of kernel opening
        GranulaityMesure_cp(start_kernel = parameters['start_kernel'],end_karnel = parameters['end_karnel'],kernel_size = parameters['kernel_size'],
                            extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],
                            videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                            Image_name = parameters['Image_name'], outPath = parameters['outPath']) # save video form cell segmentation
    elif option == "cp_gran_table_gen":
        GranularityDataGen_cp(kernelGran = parameters['kernelGran'],w = parameters['w'],h = parameters['h'],
                            extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],
                            videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                            Image_name = parameters['Image_name'], outPath = parameters['outPath'])
        # Placeholder for table generation - ration tables for labeling with corresponding images
    elif option == "dataLabeling":
        # Placeholder for app for labeling images
        dataPath = parameters['dataPath']
        imagePath = parameters['imagePath']
        app = image_labeling(dataPath,imagePath)
        app.run_server()
    elif option == "data_viz":
        # Placeholder for app for data vitalization after labeling
        dataPath = parameters['dataPath']
        imagePath = parameters['imagePath']
        app = data_viz(dataPath,imagePath)
        app.run_server()
    
    elif option == "modelBuild":
        # Placeholder for modelBuild option functionality
        Baysian_training(dataPath = parameters['dataPath'], imW = parameters['imW'], imH = parameters['imH'], thold = parameters['thold'],
                        extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],
                        videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                        Image_name = parameters['Image_name'], outPath = parameters['outPath'])
    
    elif option == "deployBuild":
        # Placeholder for deployBuild option functionality
        BayesDeploy(kernelGran = parameters['kernelGran'], dataPath = parameters['dataPath'], imW = parameters['imW'], imH = parameters['imH'], thold = parameters['thold'],
                    extract_pixel = parameters['extract_pixel'], resize_pixel = parameters['resize_pixel'],
                    videoName = parameters['videoName'], model_type = parameters['model_type'], channels = parameters['channels'],
                    Image_name = parameters['Image_name'], outPath = parameters['outPath'])
    else:
        print('Invalid option. Please choose a valid option.')


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Application for measuring cell diameter.')
    # Add an argument for the option selection
    parser.add_argument('-o', '--option', type=str,
                        choices=['measDia', 'cp_seg', 'cp_gran', 'modelBuild', 'deployBuild'],
                        required=True,
                        help='Select an option to run (measDia, cp_seg, cp_gran, modelBuild, deployBuild)')
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the run function with the selected option and debug mode
    run(args.option)

