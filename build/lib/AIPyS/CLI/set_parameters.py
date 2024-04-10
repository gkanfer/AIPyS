import os
import shutil
import h5py
from pathlib import Path
import numpy as np
import argparse
import pdb

# Assuming the 'default_parameters.h5' file is located directly inside the package folder
#default_parameters_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'default_parameters.h5')
default_parameters_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'parameters.h5')
# Determine the user's home configuration path
user_parameters_path = os.path.join(Path.home(), '.AIPyS', 'parameters.h5') # save the parameter file to .AIPyS folder. 

def ensure_user_parameters():
    """Ensure the user has their own copy of parameters.h5; if not, create it."""
    if not os.path.exists(user_parameters_path):
        os.makedirs(os.path.dirname(user_parameters_path), exist_ok=True)
        shutil.copy(default_parameters_path, user_parameters_path)
        print(f'Created a personal parameters file at {user_parameters_path}')


def update_auto_parameters(parameter_updates,user_parameters_path):
    '''
    this function is used for updating h5 file based on results
    e.g. intercept and slope from model build
    '''
    with h5py.File(user_parameters_path, 'r+') as hdf:
        for key, value in parameter_updates.items():
            if key in hdf:
                value = 'None' if value is None else value
                if isinstance(value, str):
                    hdf[key][()] = np.array(value, dtype=h5py.special_dtype(vlen=str))
                else:
                    hdf[key][...] = value
            else:
                # If the key does not exist in the HDF5 file, create a new dataset.
                if isinstance(value, str):
                    dt = h5py.special_dtype(vlen=str)
                    hdf.create_dataset(key, data=np.array(value, dtype=dt), dtype=dt)
                else:
                    hdf.create_dataset(key, data=value)
                print(f"'{key}' added to or updated in the HDF5 file with value: {value}")



def update_user_parameters(parameter_updates):
    """Example function to update parameters in the user's parameters.h5 file."""
    ensure_user_parameters()  # Ensure the user copy exists
    #pdb.set_trace()
    with h5py.File(user_parameters_path, 'r+') as hdf:
        for key, value in parameter_updates.items():
            if key in hdf:
                value = 'None' if value is None else value
                if isinstance(value, str):
                    hdf[key][()] = np.array(value, dtype=h5py.special_dtype(vlen=str))
                else:
                    hdf[key][...] = value
            else:
                # If the key does not exist in the HDF5 file, create a new dataset.
                if isinstance(value, str):
                    dt = h5py.special_dtype(vlen=str)
                    hdf.create_dataset(key, data=np.array(value, dtype=dt), dtype=dt)
                else:
                    hdf.create_dataset(key, data=value)
                print(f"'{key}' added to or updated in the HDF5 file with value: {value}")

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

# def read_user_parameters():
#     """Example function to read parameters from the user's parameters.h5 file."""
#     ensure_user_parameters()  # Ensure the user copy exists
#     with h5py.File(user_parameters_path, 'r') as hdf:
#         # Read and return parameters as needed

def main():
    parser = argparse.ArgumentParser(description='Update or add parameters and display the content of the parameters.h5 file. This script allows users to manage and view parameters stored in a .h5 file. Users can update parameter values or simply display the current configuration.', formatter_class=argparse.RawTextHelpFormatter)
    # Adding detailed help messages for each argument
    parser.add_argument('--data_dir', type=str, help="Directory where data is stored.\nThis path is used to locate your data files for processing.")
    parser.add_argument('--files', type=str, help="Path to the images or files.\nSpecify a single file or a directory containing multiple files.")
    parser.add_argument('--videoName', type=str, help="Name of the output video file.\nDetermine the filename for the processed output video.")
    parser.add_argument('--Image_name', type=str, help="Name of the image file.\nUsed to specify a particular image file name for operations.")
    parser.add_argument('--outPath', type=str, help="Output directory path.\nDefines where to store output files.")
    parser.add_argument('--block_size_cyto', type=int, help="Block size for cytoplasm segmentation.\nAn integer value affecting segmentation precision.")
    parser.add_argument('--offset_cyto', type=int, help="Offset applied to block size in cytoplasm segmentation.\nAdjusts the segmentation algorithm's sensitivity.")
    parser.add_argument('--global_ther', type=float, help="Global threshold value.\nUsed in various thresholding operations; a float between 0 and 1.")
    parser.add_argument('--clean', type=int, help="Parameter for cleaning the image.\nSpecify the intensity of the cleaning operation; higher values for more aggressive cleaning.")
    parser.add_argument('--channel', type=int, help="Channel of the image to be processed.\nIndicates which color channel to use in multi-channel image files.")
    parser.add_argument('--bit8', type=int, help="Bit depth to be considered.\nChoose between 8, 16, or 32-bit processing.")
    parser.add_argument('--ci', type=int, help="Color intensity.\nAdjusts the intensity level for color processing.")
    parser.add_argument('--start_kernel', type=int, help="Initial kernel size for operations that use a kernel.\nDetermines the starting size of the kernel in algorithms that dynamically adjust it.")
    parser.add_argument('--end_karnel', type=int, help="Final kernel size for adaptive algorithms.\nSets the maximum allowable kernel size.")
    parser.add_argument('--kernel_size', type=int, help="Kernel size for processing.\nSpecifies the fixed size of the kernel used in operations like filtering.")
    parser.add_argument('--outputImageSize', type=int, help="Output image size in pixels.\nControls the dimensions of the output image file.")
    parser.add_argument('--windowSize', type=int, help="Size of the window for image processing.\nDefines the window size for operations that process the image in sections.")
    parser.add_argument('--rnadomWindows', type=int, help="Number of random windows to be considered in processing.\nSpecifies how many random windows the algorithm should use.")
    parser.add_argument('--kernelGran', type=int, help="Granularity of the kernel.\nFine-tunes the granularity for kernel-based operations.")
    parser.add_argument('--w', type=int, help="Width of the image in pixels.\nUsed to specify the target width for image resizing operations.")
    parser.add_argument('--h', type=int, help="Height of the image in pixels.\nSets the target height for image adjustments.")
    parser.add_argument('--diameter', type=int, help="cellpose parameter.\nEstimation of cell diameter.")
    parser.add_argument('--model_type',type=str, help=" <cyto> or model_type=<nuclei>")
    parser.add_argument('--channels',type=str,help = "grayscale cellpose")
    parser.add_argument('--extract_pixel',type=int,help = "size of extraction according to mask (e.g. 50 pixel)")
    parser.add_argument('--resize_pixel',type=int,help = "resize for preforming tf prediction (e.g. 150 pixel)")
    parser.add_argument('--dataPath',type=str,help = "\imageSequence\data")
    parser.add_argument('--imagePath',type=str,help = "\imageSequence\images")
    parser.add_argument('--imW',type=int,help = "10")
    parser.add_argument('--imH',type=int,help = "5")
    parser.add_argument('--thold',type=float,help = "0.5")
    parser.add_argument('--imagesN',type=int,help = "number of images for analyse")
    parser.add_argument('--trainingDataPath',type=str,help = "use all the images in the base_training_directory")
    parser.add_argument('--areaSel',type=int,help = "area cutoff")
    parser.add_argument('--fractionData',type=int,help = "use fraction of the data for training")
    parser.add_argument('--intercept',type=float,help = "intercept estimated by the bayesian model")
    parser.add_argument('--slope',type=float,help = "intercept estimated by the bayesian model")
    
    # Argument for displaying the parameters without making updates
    parser.add_argument('--display', action='store_true', help="Display the contents of the parameters file without making any updates.\nUse this option to view the current settings stored in the .h5 file.")

    
    args = parser.parse_args()
   
    # Construct the dictionary from provided CLI arguments
    # Exclude 'h5_file' from update_dict, as it's not a parameter to store but the file path
    update_dict = {k: v for k, v in vars(args).items() if k != 'h5_file' and v is not None}
    #pdb.set_trace()
    # Update or add parameters in the H5 file
    update_user_parameters(parameter_updates = update_dict)


if __name__ == "__main__":
    main()

