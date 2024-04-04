import os
from pathlib import Path
import argparse
import subprocess
import sys
import h5py
import numpy as np

def unpack_dataset(dataset):
    """Handle decoding of datasets from HDF5, with special handling for arrays of strings."""
    if isinstance(dataset, np.ndarray) and dataset.dtype.kind == 'O':  # Object arrays, potentially strings
        return dataset[0].decode('utf-8')  # Decode each string from bytes
    elif isinstance(dataset, np.ndarray) and dataset.dtype.type is np.bytes_:  # Direct array of bytes
        return np.char.decode(dataset, 'utf-8')  # Decoding bytes array to strings
    elif isinstance(dataset, bytes):  # Single item as bytes
        return dataset.decode('utf-8')  # Decode single byte string to normal string
    else:
        return dataset  # Return as is for other datatypes


def check_and_prompt_parameters(option, h5filepath):
    required_params = {
        "measDia": ["Image_name"],
        "cp_seg_video": ["diameter","data_dir","imagesN", "Image_name", "videoName", "model_type", "channels", "outPath"],
        "cp_gran_video": ["outputImageSize","start_kernel", "end_karnel", "kernel_size", "extract_pixel", "resize_pixel", "videoName", "model_type", "channels", "Image_name", "outPath","diameter"],
        "cp_gran_table_gen": ["kernelGran", "w", "h", "extract_pixel", "resize_pixel", "videoName", "model_type", "channels", "Image_name", "outPath","imagesN","data_dir","trainingDataPath","diameter"],
        "dataLabeling": ["dataPath", "imagePath"],
        "data_viz": ["dataPath", "imagePath"],
        "modelBuild": ["diameter","dataPath", "imW", "imH", "thold", "extract_pixel", "resize_pixel","model_type", "channels", "Image_name", "outPath","areaSel","fractionData"],
        "deployBuild": ["diameter","kernelGran", "dataPath", "imW", "imH", "thold", "extract_pixel", "resize_pixel","model_type", "channels", "Image_name", "outPath","intercept","slope","areaSel","fractionData"]
    }
    
    
    # Opening the HDF5 file
    with h5py.File(h5filepath, 'r') as h5file:
        sub_data_dict = {}
        non_counts = 0
        missingParam = [] 
        for subkey in required_params[option]:
            try:
                value = unpack_dataset(h5file[subkey][()])
                sub_data_dict[subkey] = value
            except KeyError:
                raise f" {subkey} is required. use set_parameters --{subkey} to update parameters"
            # test if the parameter is missing
            if value=='None':
                non_counts =+ 1
                missingParam.append(subkey)
                # raise f" {subkey} is required. use set_parameters --{subkey} to update parameters"
    return sub_data_dict,non_counts,missingParam