import os
import numpy as np
import tifffile as tfi
from skimage.util import img_as_ubyte
import string
import random
import skimage

from AIPyS.cellpose import AIPS_cellpose as AC
from AIPyS.supportFunctions import AIPS_granularity as ag
from AIPyS.supportFunctions import AIPS_file_display as afd


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def BayesianGranularityDeploy(file,path,kernel_size,trace_a,trace_b,thold,pathOut,clean,saveMerge=False):
    '''
    on the fly cell call function for activating cells

    Parameters
    ----------
    file: str
        single channel target image
    path: str
    kernel_size: int
    trace_a: int
    trace_b: int
    thold: int
        probability threshold for calling cells
    pathOut: str
    clean: int
        remove object bellow the selected area size
    saveMerge: boolean

    Returns
    -------
    binary mask for activating the called cell
    '''
    if isinstance(clean, int) == False:
        mesg = "area size is not of type integer"
        raise ValueError(mesg)
    AIPS_pose_object = AC.AIPS_cellpose(Image_name=file, path=path, model_type="cyto", channels=[0, 0], clean = clean)
    img = AIPS_pose_object.cellpose_image_load()
    # create mask for the entire image
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
    if len(table)<5:
        with open(os.path.join(pathOut, 'cell_count.txt'), 'r') as f:
            prev_number = f.readlines()
        new_value = int(prev_number[0]) + len(table)
        with open(os.path.join(pathOut, 'cell_count.txt'), 'w') as f:
            f.write(str(new_value))
        with open(os.path.join(pathOut, 'count.txt'), 'w') as f:
            f.write(str(len(table)))
    else:
        gran = ag.GRANULARITY(image=img, mask=mask)
        granData = gran.loopLabelimage(start_kernel=1, end_karnel=kernel_size, kernel_size=kernel_size, deploy = True)
        granDataFinal = ag.MERGE().calcDecay(granData, kernel_size)
        def classify(n, thold):
            mu = trace_a + trace_b * n
            prob = 1 / (1 + np.exp(-mu))
            return prob, prob > thold
        rate = granDataFinal.intensity.values
        prob, prediction = classify(rate, thold)
        table["predict"] = prob
        image_blank = np.zeros_like(img)
        binary, table_sel = AIPS_pose_object.call_bin(table_sel_cor=table, threshold=0.9, img_blank=image_blank)
        img_gs = img_as_ubyte(binary)
        if saveMerge:
            table['predict'] = np.round(table.predict.values, 2)
            maskKeep = AIPS_pose_object.keepObject(table = table_sel)
            compsiteImage = afd.Compsite_display(input_image=img, mask_roi=maskKeep)
            LabeldImage = compsiteImage.display_image_label(table=table, font_select="arial.ttf", font_size=14,label_draw='predict', intensity=1)
            LabeldImage.save(os.path.join(pathOut, id_generator() + '.png'))
        with open(os.path.join(pathOut, 'active_cell_count.txt'), 'r') as f:
            prev_number_active = f.readlines()
        new_value = int(prev_number_active[0]) + len(table_sel)
        with open(os.path.join(pathOut, 'active_cell_count.txt'), 'w') as f:
            f.write(str(new_value))
        if os.path.exists(os.path.join(pathOut, 'binary.jpg')):
            os.remove(os.path.join(pathOut, 'binary.jpg'))
        skimage.io.imsave(os.path.join(pathOut, 'binary.jpg'), img_gs)
        #tfi.imsave(os.path.join(pathOut, 'binary.jpg'), img_gs)
        with open(os.path.join(pathOut, 'cell_count.txt'), 'r') as f:
            prev_number = f.readlines()
        new_value = int(prev_number[0]) + len(rate)
        with open(os.path.join(pathOut, 'cell_count.txt'), 'w') as f:
            f.write(str(new_value))
        with open(os.path.join(pathOut, 'count.txt'), 'w') as f:
            f.write(str(len(table)))






