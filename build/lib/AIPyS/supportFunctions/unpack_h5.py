import h5py
import numpy as np

def unpack_dataset(dataset):
    """Handle decoding of datasets from HDF5, with special handling for arrays of strings."""
    if isinstance(dataset, np.ndarray) and dataset.dtype.kind == 'O':  # Object arrays, potentially strings
        return np.array([str(item[()], 'utf-8') for item in dataset])  # Decode each string from bytes
    elif isinstance(dataset, np.ndarray) and dataset.dtype.type is np.bytes_:  # Direct array of bytes
        return np.char.decode(dataset, 'utf-8')  # Decoding bytes array to strings
    elif isinstance(dataset, bytes):  # Single item as bytes
        return dataset.decode('utf-8')  # Decode single byte string to normal string
    else:
        return dataset  # Return as is for other datatypes

def read_user_parameters(parameters_path):
    """Extract datasets, especially handling lists of strings."""
    data = {}
    with h5py.File(parameters_path, 'r') as hdf:
        for name in hdf:
            dataset = hdf[name]
            # For datasets directly reading the values
            data[name] = unpack_dataset(dataset)

    return data