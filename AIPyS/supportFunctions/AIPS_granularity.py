import numpy as np
from skimage import measure, morphology
from skimage.transform import resize
import pandas as pd
from random import randint

class GRANULARITY():
    def __init__(self,mask,image):
        self.mask = mask
        self.image = image

    def outlierRemoval(self):
        pass

    def measure_properties(self, input_image, input_mask, prop_names):
        '''
        :param: input_image (img), 8 bit greyscale image
        :param: mask (img) 32bit mask
        :param: prop_names (list), features array
        :return: skit image property table propriety table
        '''

        def sd_intensity(regionmask, intensity_image):
            return np.std(intensity_image[regionmask])

        def pixelcount(regionmask):
            return np.sum(regionmask)

        def mean_int(regionmask, intensity_image):
            return np.mean(intensity_image[regionmask])

        if prop_names is None:
            raise ValueError("features array is missing or not of type list")

        table_prop = measure.regionprops_table(
            label_image=input_mask, intensity_image=input_image, properties=prop_names,
            extra_properties=(sd_intensity, pixelcount, mean_int))
        return table_prop

    def featuresTable(self,features =["label"]):
        '''
        :param: features (list), could be a list or tuple
        :return: skit image property table and pandas DF of image intensity at the smallest kernel
        '''
        prop_names = (
            "label",
            "centroid",
            "area",
            "eccentricity",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "inertia_tensor",
            "inertia_tensor_eigvals",
            "moments",
            "moments_central",
            "moments_hu",
            "moments_normalized",
            "orientation",
            "perimeter",
            "perimeter_crofton",
            "slice",
            "solidity")
        # incase fetures are type of list then no need to do the loop
        if not(features[0] in prop_names):
            mesg = "select from the following features:{}".format(prop_names)
            raise ValueError(mesg)
        featuresSel = [feature for feature in prop_names if feature in features]
        tableOut = pd.DataFrame(self.measure_properties(input_image = self.image,input_mask = self.mask ,prop_names = featuresSel))
        # initiate table signal time zero, first column per label
        tableInit = pd.DataFrame(tableOut['mean_int'].tolist())
        return tableOut, tableInit

    def openingOperation(self,kernel):
        '''
        :param: kernel (int), size of filter kernel
        :return: opening operation image
        '''
        selem = morphology.disk(kernel, dtype=bool)
        eros_pix = morphology.erosion(self.image, footprint=selem)
        imageOpen = morphology.dilation(eros_pix, footprint=selem)
        return imageOpen

    def loopLabelimage(self,start_kernel = 2 , end_karnel = 80, kernel_size=20 , fetureLabel = ["label"], deploy = False ):
        '''
        :param start_kernel (int), smallest kernel size
        :param end_karnel (int), largest kernel size
        :param kernel_size (int), number of opening operation to apply
        :param fetureLabel (list), list of features measure per cell (beside intensity), e.g. ["label","centroid",
            "area"] default ["label"]
        :return: dataframe of image intensity per segmented cell per image
        '''
        # returns column wise table
        tableInit, tableColumn = self.featuresTable(features =fetureLabel)
        open_vec = np.linspace(start_kernel, end_karnel, kernel_size, endpoint=True, dtype=int)
        if deploy:
            limit = 1
            open_vec = [open_vec[0],open_vec[-1]]
        else:
            limit = 3
        if len(open_vec) < limit:
            mesg = "increase kernel size"
            raise ValueError(mesg)
        for opning in open_vec:
            openImage = self.openingOperation(kernel = opning)
            tableGran_temp = self.measure_properties(input_image =openImage,input_mask = self.mask ,prop_names = fetureLabel)
            tableColumn[opning] = tableGran_temp['mean_int'].tolist()
        table_ = pd.DataFrame({'label':  tableInit.index.values})
        # convert pd to numpy preform normalisation by dividing
        kernel_array = [col for col in tableColumn.columns]
        mat = tableColumn.to_numpy(copy=True)
        imageSignalZero = mat[:, 0]
        tableColumnMatNorm = mat/imageSignalZero[:, np.newaxis]
        tableColumnNorm = pd.DataFrame(tableColumnMatNorm, columns=kernel_array)
        # adding label column
        table_gran_comp = pd.concat([table_, tableColumnNorm],axis = 1)
        table_gran_comp = table_gran_comp.melt(id_vars=["label"])
        tableFinal = table_gran_comp.sort_values(['label']).reset_index(drop=True)
        tableFinal = tableFinal.rename(
            columns={"label": "labeledCell", "variable":"kernel", "value": "intensity"})
        # cellLabel intoencoder
        tableFinal['labeledCell'] = pd.Categorical(tableFinal['labeledCell'], ordered=False)
        return tableFinal

    def stackTOobjects(self, extract_pixel, resize_pixel, img_label):
        '''
        fnction similar to the EBimage stackObjectsta, return a crop size based on center of measured mask
        :param: extract_pixel (int), size of extraction acording to mask (e.g. 50 pixel)
        :param: resize_pixel (int), resize for preforming tf prediction (e.g. 150 pixel)
        :param: img_label (img), the mask value for stack
        :return: (img), center image with out background
        '''
        img = self.image
        mask = self.mask
        table = pd.DataFrame(self.measure_properties(input_image = img, input_mask = mask ,prop_names = ["centroid"]))
        # table = table.astype({"centroid-0": 'int', "centroid-1": 'int'})
        x, y = table.loc[img_label, ["centroid-0", "centroid-1"]]
        x, y = int(x), int(y)
        mask_value = mask[x, y]
        x_start = x - extract_pixel
        x_end = x + extract_pixel
        y_start = y - extract_pixel
        y_end = y + extract_pixel
        if x_start < 0 or x_end < 0 or y_start < 0 or y_end < 0:
            return None, None
        else:
            mask_bin = np.zeros((np.shape(img)[0], np.shape(img)[1]), np.int32)
            mask_bin[mask == mask_value] = 1
            masked_image = img * mask_bin
            stack_img = masked_image[x_start:x_end, y_start:y_end]
            stack_img = resize(stack_img, (resize_pixel, resize_pixel), anti_aliasing=False)
            return stack_img, mask_value


class MERGE:
    '''
    merge granularity table
    This function is working with the glob function where the complete path is given
    '''
    @staticmethod
    def replaceLabel(arr):
        '''
        :param: arr (list), pandas series
        :return: replace label values with unique values
        '''
        uniqueLabel = np.unique(arr)
        idx = 0
        while idx <= len(uniqueLabel)-1:
            arr[arr==uniqueLabel[idx]] = randint(0, 500_000)
            idx += 1
        return arr

    def mergeTable(self,tableInput_name_list):
        '''
        :param: tableInput_name_list (list), array of csv file names (shuffled and merged)
        :return: merge all the tables
        '''
        idx = 0
        while idx <= len(tableInput_name_list)-1:
            if idx == 0:
                df = pd.read_csv(tableInput_name_list[0])
                idx += 1
                df['labeledCell'] = self.replaceLabel(df.labeledCell.values)
                imagename = tableInput_name_list[0].split("\\")[-1]
                df['Unnamed: 0'] = imagename
            else:
                df_temp = pd.read_csv(tableInput_name_list[idx])
                df_temp['labeledCell'] = self.replaceLabel(df_temp.labeledCell.values)
                imagename = tableInput_name_list[idx].split("\\")[-1]
                df_temp['Unnamed: 0'] = imagename
                df = pd.concat((df,df_temp),ignore_index=True,axis=0)
                idx += 1
        df = df.rename(columns={'Unnamed: 0':'image_name'})
        df = df.sort_values(by=['labeledCell', 'kernel'], na_position='first')
        return df

    @staticmethod
    def meanIntensity(df, group):
        '''
        :param: df (data frame) pandas Data frame
        :param: group (int) select the labeled Group
        :return: intensity percentage list and kernels
        '''
        dfSel = df.loc[df['classLabel'] == group]
        uniqueKernel = np.unique(dfSel.kernel.values)
        meanInten = []
        kernelArray = []
        for i, kernel in enumerate(uniqueKernel):
            arrTemp = dfSel.loc[dfSel['kernel'] == kernel, 'intensity'].to_list()
            # clean array
            arrTempClean = [arr for arr in arrTemp if str(arr) != 'nan']
            meanInten.append(sum(arrTempClean) / len(arrTemp))
            kernelArray.append(kernel)
        return meanInten, kernelArray

    @staticmethod
    def cleanTable(df):
        '''
        :param: df (data frame), pandas Data frame
        :param: group (int), selected labeled Group
        :return: remove null observations
        '''
        dfClean = df.copy()
        uniqeImageLabel = np.unique(df.labeledCell.values)
        for imagename in uniqeImageLabel:
            df_temp = df.loc[df['labeledCell']==imagename,:]
            arr = df_temp.intensity.values.tolist()
            arr.pop(0)
            arrTemp = np.mean(arr)
            if not(arrTemp>0):
                idx = dfClean.loc[df['labeledCell']==imagename,:].index
                dfClean.drop(idx, inplace=True)
        return dfClean



    @staticmethod
    def calcDecay(df,kernelSelect):
        '''
        :param: df (data frame), pandas finel table
        :param: kernelSelect (int) kernel size selected
        :return: filter table for the kernel selcted result in 50% reduction in signal
        '''
        return df.loc[df['kernel']==kernelSelect,:]





