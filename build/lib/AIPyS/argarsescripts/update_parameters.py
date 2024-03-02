import h5py
import numpy as np

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
    "channels":[0,0],
    "extract_pixel":50,
    "resize_pixel":150,
    "dataPath":None,
    "imagePath":None,
    "imW":10,
    "imH":10,
    "thold":0.5,
    }

# Specify the HDF5 file name
filename = 'parameters.h5'

# Open a new HDF5 file
with h5py.File(filename, 'w') as hdf:
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

print(f"Parameters saved to {filename}")

# # Open the file in 'r+' mode, allowing you to read and write
# with h5py.File(filename, 'r+') as hdf:
#     # Specify the dataset you wish to update and the new value
#     key_to_update = "block_size_cyto"
#     new_value = 5  # Make sure the new value is compatible with the dataset

#     # Check if the dataset exists in the file
#     if key_to_update in hdf:
#         # Update the dataset with the new value
#         hdf[key_to_update][...] = new_value
#         print(f"Updated '{key_to_update}' with the new value: {new_value}")
#     else:
#         print(f"Dataset '{key_to_update}' not found.")
