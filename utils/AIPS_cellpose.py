import numpy as np
import time, os, sys
from urllib.parse import urlparse
import skimage.io
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib as mpl
from urllib.parse import urlparse
from cellpose import models, core
from cellpose.io import logger_setup
from cellpose import utils
import glob
import pandas as pd
from scipy.stats import skew

from PIL import Image, ImageEnhance, ImageDraw,ImageFont
from skimage import io, filters, measure, color, img_as_ubyte
from skimage.draw import disk
from skimage import measure, restoration,morphology

import seaborn as sns

from utils import AIPS_granularity as ag
from utils import AIPS_file_display as afd
from utils import AIPS_cellpose as AC

def granularityMesure_cellpose(file,path,classLabel,outPath, clean = None, outputTableName = None,):
    '''
    function description:
        1) cell segmented using cellpose
        2) clean data based on cell area detected
        3) granularity measure
    output:
        1) Segmented image composition with area label
        2) Area histogram plot
        3) Segmented image composition after removal of small objects
        4) plot of Mean intensity over opening operation (granularity spectrum)
    :parameter
    clean: int, remove object bellow the selected area size
    classLabel: int, assign label
    file: str
    path: str
    outPath: str
    outputTableName: str, e.g. "outputTableNorm.csv"


    Note: required single channel tif
    '''
    AIPS_pose_object = AC.AIPS_cellpose(Image_name=file, path=path, model_type="cyto", channels=[0, 0])
    img = AIPS_pose_object.cellpose_image_load()
    mask, table = AIPS_pose_object.cellpose_segmantation(image_input=img)
    compsite = afd.Compsite_display(input_image=img, mask_roi=mask)
    compsiteImage  = compsite.display_image_label(table=table, font_select="arial.ttf", font_size=24,  intensity=2,label_draw = 'area')
    compsiteImage.save(os.path.join(outPath,"merge.png"), "PNG")
    if clean:
        objectidx = table.loc[table['area'] < clean,:].index.tolist()
        mask, table = AIPS_pose_object.removeObjects(objectList=objectidx)
        compsite = afd.Compsite_display(input_image=img, mask_roi=mask)
        compsiteImage = compsite.display_image_label(table=table, font_select="arial.ttf", font_size=24, intensity=2,label_draw='area')
        compsiteImage.save(os.path.join(outPath,"mergeClean.png"), "PNG")
    gran = ag.GRANULARITY(image=img, mask=mask)
    granData = gran.loopLabelimage(start_kernel=2, end_karnel=7, kernel_size=7)
    granOriginal, _ = gran.featuresTable(features=['label', 'centroid'])
    granData["classLabel"] = classLabel
    if outputTableName is None:
        granData.to_csv(os.path.join(outPath,'granularity.csv'))
    else:
        granData.to_csv(os.path.join(outPath, outputTableName))
    Intensity, Kernel = ag.MERGE().meanIntensity(granData, group=classLabel)
    df = pd.DataFrame({"kernel":Kernel,"Signal intensity (ratio)":Intensity})
    from matplotlib.backends.backend_pdf import PdfPages
    def generate_plots():
        def hist():
            fig, ax = plt.subplots()
            sns.histplot(data=table, x='area', kde=True, color=sns.color_palette("Set2")[1], binwidth=50).set(title = 'Cell area distribution')
            return ax
        def line():
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x="kernel", y="Signal intensity (ratio)").set(title='Granularity spectrum plot')
            return ax
        plot1 = hist()
        plot2 = line()
        return (plot1, plot2)

    def plots2pdf(plots, fname):
        with PdfPages(fname) as pp:
            for plot in plots:
                pp.savefig(plot.figure)
    plots2pdf(generate_plots(), os.path.join(outPath,'outPlots.pdf'))



class AIPS_cellpose:
    def __init__(self, Image_name=None, path=None, image = None, mask = None, table = None, model_type = None, channels = None, clean = None ):
        '''
        :param Image_name: str
        :param path: str
        :param image: inputimage for segmantion
        :param model_type: 'cyto' or model_type='nuclei'
        :param clean: int, remove object bellow the selected area size
        :param channels: # channels = [0,0] # IF YOU HAVE GRAYSCALE
                    channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
                    channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

                    or if you have different types of channels in each image
                    channels = [[2,3], [0,0], [0,0]]
                    channels = [1,1]

        '''
        self.Image_name = Image_name
        self.path = path
        self.image = image
        self.mask = mask
        self.table = table
        self.model_type = model_type
        self.channels = channels
        self.clean = clean


    def cellpose_image_load(self):
        ''':parameter
        Image: File name (tif format) - should be greyscale
        path: path to the file
        :return
        grayscale_image_container: dictionary of np array
        '''
        self.image = skimage.io.imread(os.path.join(self.path,self.Image_name))
        return self.image

    def cellpose_segmantation(self, image_input):
        use_GPU = core.use_gpu()
        model = models.Cellpose(gpu=use_GPU, model_type=self.model_type)
        self.mask, flows, styles, diams = model.eval(image_input, diameter=None, flow_threshold=None, channels=self.channels)
        self.table = pd.DataFrame(
            measure.regionprops_table(
                self.mask,
                intensity_image=image_input,
                properties=['area', 'label', 'centroid'])).set_index('label')
        if self.clean:
            if isinstance(self.clean, int)==False:
                mesg = "area size is not of type integer"
                raise ValueError(mesg)
            objectidx = self.table.loc[self.table['area'] < self.clean, :].index.tolist()
            self.mask, self.table = self.removeObjects(objectList=objectidx)
        return self.mask, self.table

    def stackObjects_cellpose_ebimage_parametrs_method(self, image_input ,extract_pixel, resize_pixel, img_label):
        '''
        fnction similar to the EBimage stackObjectsta, return a crop size based on center of measured mask
        :param extract_pixel: size of extraction acording to mask (e.g. 50 pixel)
        :param resize_pixel: resize for preforming tf prediction (e.g. 150 pixel)
        :param img_label: the mask value for stack
        :return: center image with out background
        '''
        img = image_input
        mask= self.mask
        table = self.table
        #table = table.astype({"centroid-0": 'int', "centroid-1": 'int'})
        x, y = table.loc[img_label, ["centroid-0", "centroid-1"]]
        x, y = int(x), int(y)
        mask_value = mask[x, y]
        x_start = x - extract_pixel
        x_end = x + extract_pixel
        y_start = y - extract_pixel
        y_end = y + extract_pixel
        if x_start < 0 or x_end < 0 or y_start < 0 or y_end < 0:
            stack_img = None
            mask_value = None
            return stack_img, mask_value
        else:
            mask_bin = np.zeros((np.shape(img)[0], np.shape(img)[1]), np.int32)
            mask_bin[mask == mask_value] = 1
            masked_image = img * mask_bin
            stack_img = masked_image[x_start:x_end, y_start:y_end]
            stack_img = skimage.transform.resize(stack_img, (resize_pixel, resize_pixel), anti_aliasing=False)
            return stack_img, mask_value

    def measure_properties(self, input_image):
        def sd_intensity(regionmask, intensity_image):
            return np.std(intensity_image[regionmask])

        def skew_intensity(regionmask, intensity_image):
            return skew(intensity_image[regionmask])

        def pixelcount(regionmask):
            return np.sum(regionmask)

        def mean_int(regionmask, intensity_image):
            return np.mean(intensity_image[regionmask])

        prop_names = [
            "label",
            "centroid",
            "area",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "inertia_tensor",
            "inertia_tensor_eigvals",
            # "moments",
            # "moments_central",
            # "moments_hu",
            # "moments_normalized",
            "orientation",
            "perimeter",
            "perimeter_crofton",
            # "slice",
            "solidity"
        ]
        table_prop = measure.regionprops_table(
            self.mask, intensity_image=input_image, properties=prop_names,
            extra_properties=(sd_intensity, skew_intensity, pixelcount, mean_int)
        )
        return table_prop

    def display_image_prediction(self,img ,prediction_table, font_select = "DejaVuSans.ttf", font_size = 4, windows=False, lable_draw = 'predict',round_n = 2):
        '''
        ch: 16 bit input image
        mask: mask for labale
        lable_draw: 'predict' or 'area'
        font_select: copy font to the working directory ("DejaVuSans.ttf" eg)
        font_size: 4 is nice size
        round_n: integer how many number after decimel

        return:
        info_table: table of objects measure
        PIL_image: 16 bit mask rgb of the labeled image
        '''
        # count number of objects in nuc['sort_mask']
        img_gs = img_as_ubyte(img)
        PIL_image = Image.fromarray(img_gs)
        # round
        info_table = prediction_table.round({'centroid-0': 0, 'centroid-1': 0})
        info_table['predict_round'] = info_table.loc[:, 'predict'].astype(float).round(round_n)
        info_table['area_round'] = info_table.loc[:, 'area'].astype(float).round(round_n)
        info_table = info_table.reset_index(drop=True)
        draw = ImageDraw.Draw(PIL_image)
        if lable_draw ==  'predict':
            lable = 'predict_round'
        else:
            lable = 'area_round'
        # use a bitmap font
        if windows:
            font = ImageFont.truetype("arial.ttf", font_size, encoding="unic")
        else:
            font = ImageFont.truetype(font_select, font_size)
        for i in range(len(info_table)):
            draw.text((info_table.loc[i, 'centroid-1'].astype('int64'), info_table.loc[i, 'centroid-0'].astype('int64')),
                      str(info_table.loc[i, lable]), 'red', font=font)
        return info_table, PIL_image

    def call_bin(self,table_sel_cor, threshold ,img_blank):
        '''
        :parameter:
        table_sel_cor: pandas table contain the center coordinates
        threshold: thershold for predict phenotype (e.g. 0.5)
        img_blank: blank image in the shape of the input image
        :return: binary image of the called masks, table_sel
        '''
        table_na_rmv_trgt = table_sel_cor.loc[table_sel_cor['predict'] > threshold, :]
        for label in table_na_rmv_trgt.index.values:
            x, y = table_na_rmv_trgt.loc[label,["centroid-0", "centroid-1"]].tolist()
            row, col = disk((int(x), int(y)), 10)
            img_blank[row, col] = 1
        return img_blank, table_na_rmv_trgt

    def removeObjects(self,objectList):
        '''
        :parameter:
        objectList: list of objects to remove
        :return:
        update mask and table
        '''
        if objectList is None:
            raise ValueError("Object list is missing")
        # check whether the selected object included in objectList
        tableList = self.table.index.values.tolist()
        match = 0
        for object in objectList:
            if object in tableList:
                match += 1
        if match==0:
            return self.mask, self.table
        else:
            for object in objectList:
                self.mask[self.mask==object] = 0
            self.table.drop(objectList, inplace=True)
            return self.mask, self.table

    def keepObject(self, table):
        '''
        :param table: keep all the object which are predicted above the threshold
        :return: ROI image mask of selected objects
        '''
        nmask = np.zeros_like(self.mask)
        for label in table.index.values:
            nmask[self.mask == label] = label
        return nmask












