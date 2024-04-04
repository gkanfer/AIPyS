import os
import shutil
import h5py
from pathlib import Path
import numpy as np
import argparse
import re
import pdb
from pathlib import Path

def main(path):
    newpath = path.encode('utf-8')
    print(newpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update or add parameters and display the content of the parameters.h5 file. This script allows users to manage and view parameters stored in a .h5 file. Users can update parameter values or simply display the current configuration.', formatter_class=argparse.RawTextHelpFormatter)
    # Adding detailed help messages for each argument
    parser.add_argument('--data_dir', type=str, help="Directory where data is stored.\nThis path is used to locate your data files for processing.")
    args = parser.parse_args()
    main(args.data_dir)
    