import h5py
import numpy as np
import os
import shutil
from pathlib import Path
import argparse

# Define your parameters
params = {
    "data_dir": None,
    "files": None,
    "videoName": None,
    "Image_name": None,
    "outPath": None,
    "block_size_cyto": 3,
    "offset_cyto": -5,
    "global_ther": 0.51,
    "clean": 3,
    "channel": 0,
    "bit8": 20,
    "ci": 3,
    "start_kernel": 2,
    "end_karnel": 50, 
    "kernel_size": 20,
    "outputImageSize": 500,
    "windowSize": 6,
    "rnadomWindows": 10,
    "kernelGran": 4,
    "w": 500,
    "h": 500,
    "diameter":50,
    "model_type":'cyto',
    "channels":'grayscale',
    "extract_pixel":50,
    "resize_pixel":150,
    "dataPath":None,
    "imagePath":None,
    "imW":10,
    "imH":10,
    "thold":0.5,
    "imagesN":30,
    "trainingDataPath":None,
    "areaSel":-1,
    "fractionData":-1,
    "intercept":0.0,
    "slope":0.0
    }

# Specify the HDF5 file name
user_parameters_path = os.path.join(Path.home(), '.AIPyS', 'parameters.h5')
dirPath = os.path.join(Path.home(), '.AIPyS') 

def parametersGnert():
    if not os.path.exists(dirPath):
       os.makedirs(os.path.dirname(dirPath), exist_ok=True) 
    # Open a new HDF5 file
    with h5py.File(user_parameters_path, 'w') as hdf:
        # Iterate through dictionary items and save them to the HDF5 file
        for key, value in params.items():
            # Convert None values to a recognizable string, as NoneType can't be directly stored
            if value is None:
                value = 'None'
            # Check if value needs to be stored as a string
            if isinstance(value, str):
                dt = h5py.special_dtype(vlen=str)  # special dtype for variable-length strings
                dset = hdf.create_dataset(key, (1,), dtype=dt)
                dset[0] = value
            else:
                # Directly store if the value is numeric
                hdf.create_dataset(key, data=value)
    print(f"Parameters saved to {user_parameters_path}")

def resetParameters():
    parametersGnert()
    print(f'Done')

def display_parameters():
    """Read and display parameters from the user's parameters.h5 file."""
    with h5py.File(user_parameters_path, 'r') as hdf:
        print("Current parameters stored in the H5 file:")
        for key in hdf.keys():
            print(f"{key}: {hdf[key][()]}")

def main():
    parser = argparse.ArgumentParser(description='Manage parameters file', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('action', choices=['generate', 'reset', 'display'],
                        help="generate: Create a new parameters file with default values."
                             "reset: Reset the parameters file to default values."
                             "display: Show current values in the parameters file.")
    args = parser.parse_args()
    
    if args.action == 'generate':
        parametersGnert()
    elif args.action == 'reset':
        resetParameters()
    elif args.action == 'display':
        display_parameters()
    else:
        raise ValueError("Unknown action. Available actions: generate, reset, display")

if __name__ == "__main__":
    main()