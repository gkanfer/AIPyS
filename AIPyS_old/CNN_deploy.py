'''
    Deployment of convolutional neural network
'''
import time, os, sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tfi
from AIPyS_old import AIPS_cellpose as AC

def CNNDeploy(path_model,model,file_name,path,pathOut,areaFilter,thr=0.5):
    """
    Parameters
    ----------
    path_model: str
    model: h5
        tensor flow model file
    file_name: str
    path: str
    pathOut: str
    areaFilter: int
    thr: float
    """
    model_cnn = load_model(os.path.join(path_model, model))
    AIPS_pose_object = AC.AIPS_cellpose(Image_name=file_name, path=path, model_type="cyto", channels=[0, 0])
    img = AIPS_pose_object.cellpose_image_load()
    # create mask for the entire image
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
    table["predict"] = 'Na'
    for i in range(len(table)):
        stack, stack_v = AIPS_pose_object.stackObjects_cellpose_ebimage_parametrs_method(image_input=img,
                                                                                         extract_pixel=50,
                                                                                         resize_pixel=150,
                                                                                         img_label=table.index.values[
                                                                                             i])
        if stack is None:
            continue
        else:
            plt.imsave("temp.png", stack)
            test_imgs = img_to_array(load_img("temp.png", target_size=(150, 150)))
            test_imgs = np.array(test_imgs)
            test_imgs_scaled = test_imgs
            test_imgs_scaled /= 255
            pred = model_cnn.predict(test_imgs_scaled.reshape(1, 150, 150, 3), verbose=0).tolist()[0][0]
            table.loc[i, "predict"] = pred
    # remove nas
    table_na_rmv = table.loc[table['predict'] != 'Na', :]
    table_na_rmv = table_na_rmv.loc[table['area'] > areaFilter, :]
    ##### binary image contains the phnotype of intrse #####
    image_blank = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor=table_na_rmv, threshold=thr, img_blank=image_blank)
    tfi.imsave(os.path.join(pathOut, 'binary.tif'), binary)