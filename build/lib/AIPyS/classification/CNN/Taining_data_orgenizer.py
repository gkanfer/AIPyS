# import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from IPython.display import clear_output

# import glob
# import numpy as np
import os
import shutil
# import glob
# from sklearn.model_selection import train_test_split

np.random.seed(42)
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

'''
Handle training data for DL binary classification
1) mix all the files in one location with proper label
2) 
'''

class classification_data_orgenizer:
    # organize images files and folder for binary classification
    # creates train, validation and test directories
    def __init__(self,report = None,path_input = None, path_origen = None,
                path_class_A = None, path_class_B = None,file_name_CA = None,file_name_CB = None,label_A = None,
                 label_B = None, file_extention= None):
        '''
        path_input - directory contain mixture of images for group a and b
        path_origen - training directory
        file_name_CA - list,str, file names group A
        file_name_CB - list,str, file names group A
        label_A - str ,example: 'norm'
        label_B - str , example: 'pheno'
        file_extention - str, example: 'png' or 'tif'
        '''
        self.report = report
        self.path_input = path_input
        self.path_origen = path_origen
        self.path_class_A = path_class_A
        self.path_class_B = path_class_B
        self.file_name_CA = file_name_CA
        self.file_name_CB = file_name_CB
        self.label_A = label_A
        self.label_B = label_B
        self.file_extention = file_extention


    def get_file_names_list(self):
        file_names = [file for file in os.listdir(self.path_input)
                    if file.endswith(self.file_extention)]
        self.file_name_CA = [file_name for file_name in file_names
                        if self.label_A in file_name]
        self.file_name_CB = [file_name for file_name in file_names
                        if self.label_B in file_name]
        return self.file_name_CA, self.file_name_CB

    def split_traning_set_and_copy(self):
        train_ratio = 0.80
        #test_ratio = 0.10
        validation_ratio = 0.10
        if len(self.file_name_CA) >= len(self.file_name_CB):
            size_train = int(len(self.file_name_CB)*train_ratio)
            size_val = int(len(self.file_name_CB)*validation_ratio)
        else:
            size_train = int(len(self.file_name_CA)*train_ratio)
            size_val = int(len(self.file_name_CA)*validation_ratio)
        A_train = np.random.choice(self.file_name_CA, size=size_train, replace=False)
        B_train = np.random.choice(self.file_name_CB, size=size_train, replace=False)
        A_val_test = list(set(self.file_name_CA) - set(A_train))
        B_val_test = list(set(self.file_name_CB) - set(B_train))
        A_val = np.random.choice(A_val_test, size=size_val, replace=False)
        B_val = np.random.choice(B_val_test, size=size_val, replace=False)
        A_test = list(set(A_val_test) - set(A_val))
        B_test = list(set(B_val_test) - set(B_val))
        statment_a = 'A set datasets: {}{}{}'.format(np.shape(A_train), np.shape(A_val),np.shape(A_test))
        statment_b = 'B set datasets: {}{}{}'.format(np.shape(B_train), np.shape(B_val),np.shape(B_test))
        # copy files to the folder
        os.chdir(self.path_origen)
        train_dir = 'training_data'
        val_dir = 'validation_data'
        test_dir = 'test_data'
        train_files = np.concatenate([A_train, B_train])
        validate_files = np.concatenate([A_val, B_val])
        test_files = np.concatenate([A_test, B_test])
        os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
        os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
        os.mkdir(test_dir) if not os.path.isdir(test_dir) else None
        
        train_dir = os.path.join(self.path_origen, 'training_data')
        val_dir = os.path.join(self.path_origen, 'validation_data')
        test_dir = os.path.join(self.path_origen, 'test_data')
        
        clear_output(wait=True) # remove all display 
        os.chdir(self.path_input)
        for i,fn in enumerate(train_files):
            shutil.copy(fn, train_dir)
            print('-' * i, end = '\r')
        print('{}'.format(fn[:5]))

        for i,fn in enumerate(validate_files):
            shutil.copy(fn, val_dir)
            print('-' * i, end = '\r')
        print('{}'.format(fn[:5]))

        for i,fn in enumerate(test_files):
            shutil.copy(fn, test_dir)
            print('-' * i, end = '\r')
        print('{}'.format(fn[:5]))
        return statment_a, statment_b, train_files, validate_files, test_files











        
   