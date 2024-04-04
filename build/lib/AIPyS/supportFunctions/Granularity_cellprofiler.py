import numpy as np
from skimage import measure, morphology
from skimage.transform import resize
from skimage import io
import skimage
from skimage.util import img_as_ubyte
import pandas as pd
from random import randint
from cellpose import models, core
from skimage import img_as_ubyte
from skimage.draw import disk
from skimage import measure
from skimage import exposure
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import os
import re
import random
import string
from IPython.display import clear_output
import cv2
import glob
import pdb
import seaborn as sns
# from tifffile import imread
from AIPyS.supportFunctions import AIPS_module as am
from AIPyS.cellpose import AIPS_cellpose as AC
from AIPyS.supportFunctions.AIPS_module import AIPS_Cyto_Global
import pdb

class Granularity_cellprofiler:
    def __init__(self, image_list, diameter_cp = None, model_type="cyto",channels = [0,0]):
        '''
        image_list = glob.glob(data_dir + '*tif')
        '''
        self.image_list = image_list
        self.images = self.loadingImages()
        self.diameter_cp = diameter_cp
        self.model_type = model_type
        self.channels = channels
    
    def loadingImages(self):
        return [skimage.io.imread(image) for image in self.image_list]

    def openingOperation(self,kernel,image):
        '''
        :param: kernel (int), size of filter kernel
        :return: opening operation image
        '''
        selem = morphology.disk(kernel, dtype=bool)
        eros_pix = morphology.erosion(image, footprint=selem)
        imageOpen = morphology.dilation(eros_pix, footprint=selem)
        return imageOpen

    def cellpose_segmantation(self, image_input):
        use_GPU = core.use_gpu()
        model = models.Cellpose(gpu=use_GPU, model_type=self.model_type)
        mask, _, _ , _ = model.eval(image_input, diameter=self.diameter_cp, flow_threshold=None, channels=[0,0])
        return mask

    def openingOperation(self,image_input,kernel):
        '''
        :param: kernel (int), size of filter kernel
        :return: opening operation image
        '''
        selem = morphology.disk(kernel, dtype=bool)
        eros_pix = morphology.erosion(image_input, footprint=selem)
        imageOpen = morphology.dilation(eros_pix, footprint=selem)
        return imageOpen

    def normalize(self,image_input,mean,std):
        return (image_input - mean)/std

class ImageToVideo:
    def __init__(self,origFolder,imageNames,outFolder,alpha = 1.5, beta = 0,displayPrograss = True):
        self.origFolder = origFolder
        self.imageNames = imageNames
        self.outFolder = outFolder
        self.alpha = alpha
        self.beta = beta
        self.displayPrograss = displayPrograss
        self.loadImages()
    
    def extract_mean_sd(self,filename):
        mean = re.search('mean_(.*?)_', filename)
        if mean:
            mean = float(mean.group(1))
        sd = re.search('sd_(.*?)_', filename)
        if sd:
            sd = float(sd.group(1))
        return mean, sd

    def addText(self,image_input,text):
        return cv2.putText(image_input, f'mean: {text[0]} \n sd: {text[1]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    def rgbImge(self,image_input):
        input_gs_image = (image_input / image_input.max()) * 255
        ch2_u8 = np.uint8(input_gs_image)
        rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
        rgb_input_img[:, :, 0] = ch2_u8
        rgb_input_img[:, :, 1] = ch2_u8
        rgb_input_img[:, :, 2] = ch2_u8
        pilimage = Image.fromarray(rgb_input_img)
        return rgb_input_img
    
    def imagesTovideo(self,files,imageName):
        frame = skimage.io.imread(files[0])
        frame = self.rgbImge(frame)
        width,height,layers = frame.shape
        video = cv2.VideoWriter(os.path.join(self.outFolder,imageName + '_out.avi'), cv2.VideoWriter_fourcc(*'MJPG'),10.0,(width,height),  isColor = True)  
        for file in files:
            mean, sd = self.extract_mean_sd(file)
            frame = io.imread(file)
            frame = self.rgbImge(frame)
            # add text:
            self.addText(frame,(mean,sd))
            #frame = ((frame - frame.min()) / (frame.ptp()) * 255).astype(np.uint8)
            video.write(frame)
        video.release()
    
    def loadImages(self):
        if self.displayPrograss:
           clear_output(wait=True)
        for i,imageName in enumerate(self.imageNames):
            files = glob.glob(self.origFolder + imageName + '/ImageSplit/*tif')    
            #pdb.set_trace()
            self.imagesTovideo(files,imageName)
            if self.displayPrograss:
                print('-' * i, end = '\r')


class ImageExploration(Granularity_cellprofiler,ImageToVideo):
    '''
    Explore several objects per image and intensity to determine sliding window size.
    '''
    def __init__(self,image_list,diameter_cp):
        super().__init__(image_list,diameter_cp)

    def getImageDetails(self):
        singleImage = self.images[np.random.randint(0,len(self.images))]
        return print(f'Image size: {singleImage.shape}, /n Image type: {singleImage.dtype}')
        
    def getMeanStd(self):
        '''
        Get the images mean and standard deviation
        '''
        im_inten = []
        for i in range(len(self.images)):
            im_inten.append(np.mean(self.images[i]))
        return print(f'Image mean: {np.mean(im_inten)}, /n standard deviation: {np.std(im_inten)}')

    
    def getCellPerImage(self):
        '''
        Knowing the diameter_cp number helps for cleaning the mask. for example diameter_cp = 100. 
        return the mean number of cells per image list, helps decide running window size
        '''
        cellNum = []
        for i in range(len(self.images)):
            mask = self.cellpose_segmantation(self.images[i])
            cellNum.append(len(np.unique(mask)))
        return print(f'The mean number of cells per image: {np.mean(cellNum)}')
    
    
    def kernelLossVideo_HTML(self,start_kernel = 2 , end_karnel = 20, kernel_size=10, ci = 1):
        '''
        ci - image intensity 
        return video showing image lost
        '''
        kernels = np.linspace(start_kernel, end_karnel, kernel_size, endpoint=True, dtype=int)
        singleImage = self.images[np.random.randint(0,len(self.images))]
        images = [self.openingOperation(singleImage, kernel) for kernel in kernels]
        fig, ax = plt.subplots()

        ax.axis('off')
        
        # Display the first frame
        im = ax.imshow(images[0]*ci, cmap='gray')
        
        # Function to update figure
        def update(i):
            # Update the image data
            im.set_array(images[i]*ci)
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(images), interval=500) # 500ms between frames
        HTML(ani.to_jshtml())

    def kernelLossVideo(self,pathout = '',output_name = 'kernelLoss.avi',start_kernel = 2 , end_karnel = 20, kernel_size=10, ci = 1):
        '''
        ci - image intensity 
        return video showing image lost
        '''
        kernels = np.linspace(start_kernel, end_karnel, kernel_size, endpoint=True, dtype=int)
        files = self.image_list[np.random.randint(0,len(self.image_list))]
        singleImage = skimage.io.imread(files)
        images = [self.openingOperation(singleImage, kernel) for kernel in kernels]
        frame = self.rgbImge(images[0])
        width,height,layers = frame.shape
        video = cv2.VideoWriter(os.path.join(pathout,output_name), cv2.VideoWriter_fourcc(*'MJPG'),1.0,(width,height),  isColor = True)
        #pdb.set_trace()
        for image in images:
            mean,sd = np.mean(image), np.std(image)
            frame = self.rgbImge(image)
            # add text:
            #self.addText(frame,(mean,sd))
            video.write(frame)
        video.release()


class RunningWindow(Granularity_cellprofiler):
    '''
    Split the image according to the window size to identify the smallest image size with the highest possibility of identifying the phenotypic cell. 
    '''
    def __init__(self,image_list,windowSize,rnadomWindows):
        self.windowSize = windowSize # for example split image to 6 slides
        self.rnadomWindows = rnadomWindows # for example add 3 more random slides
        super().__init__(image_list)

    def runningWin(self,image_input):
        '''
        creates a running window sized by splitting the image to equal size images
        '''
        if not isinstance(self.windowSize, (int, float)):
                raise ValueError("window size is missing")
        if not isinstance(self.rnadomWindows, (int, float)):
            raise ValueError("rando crop generator is missing choose a number")
        minAxsis = min(image_input.shape[0],image_input.shape[1])
        winSize = self.windowSize
        while minAxsis % winSize != 0:
            minAxsis -= 1
        stepSize =  int(minAxsis/winSize)
        print(f'max image size: {minAxsis}, step size: {stepSize}')
        imageSlice = []
        for i in range(0,minAxsis,stepSize):
            for j in range(0,minAxsis,stepSize):
                imageSlice.append(image_input[i:i+stepSize,j:j+stepSize])
        # adding rando slices
        rnadomWindows = [1]*self.rnadomWindows
        range_xy = 0+stepSize,minAxsis-250
        while rnadomWindows:
            x = np.random.randint(low = range_xy[0],high = range_xy[1],size =1).tolist()[0]
            imageSlice.append(image_input[x:x+stepSize,x:x+stepSize])    
            rnadomWindows.pop()
        return imageSlice

class Granularity(RunningWindow):
    def __init__(self,pathOrig,meanInten,stdInten,kernel,image_list,windowSize,rnadomWindows,displayPrograss = True):
        self.pathOrig = pathOrig
        self.meanInten = meanInten
        self.stdInten = stdInten
        self.kernel = kernel
        self.displayPrograss = displayPrograss
        super().__init__(image_list,windowSize,rnadomWindows)

    def generate_random_string(self,length):
        letters = string.ascii_letters  # this includes both lowercase and uppercase letters
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
    
    def getMeanSD(self,input_image):
        return np.round(np.mean(input_image),3),np.round(np.std(input_image),3)
    
    def imageMeasure(self,image_input):
        meanInten = self.meanInten
        stdInten = self.stdInten
        image_norm = self.normalize(image_input,meanInten,stdInten)
        image_gran_cal = self.openingOperation(image_input = image_norm, kernel = self.kernel)
        mean,sd = self.getMeanSD(image_gran_cal)
        return mean,sd
    
    def granCalc(self):
        '''
        Save normal mean and sd intensity of normal images and crop image
        '''
        if not isinstance(self.meanInten, (int, float)):
                raise ValueError("calculate mean image intensity")
        if not isinstance(self.stdInten, (int, float)):
            raise ValueError("calculate sd image intensity")
        for image in self.images:
            # creat folder
            image_name = self.generate_random_string(7)
            tempFolder = os.path.join(self.pathOrig,image_name)
            ImageOrig = os.path.join(tempFolder,'ImageOrig')
            ImageSplit = os.path.join(tempFolder,'ImageSplit')
            os.makedirs(tempFolder,exist_ok=True)
            os.makedirs(ImageOrig,exist_ok=True)
            os.makedirs(ImageSplit,exist_ok=True)
            mean,sd = self.imageMeasure(image)
            skimage.io.imsave(os.path.join(ImageOrig,image_name + f'_mean_{mean}' + f'_sd_{sd}_' +'.tif'), image)
            imageSlice = self.runningWin(image_input = image)
            if self.displayPrograss:
                clear_output(wait=True)
                print(f'Image Name: {image_name}') 
            for i,imageCrop in enumerate(imageSlice):
                mean,sd = self.imageMeasure(imageCrop)
                skimage.io.imsave(os.path.join(ImageSplit,image_name + f'_mean_{mean}' + f'_sd_{sd}_' +'.tif'), imageCrop)
                if self.displayPrograss:
                    print('-' * i, end = '\r')

class GranulaityMesure(AIPS_Cyto_Global):
    def __init__(self,Image_name, block_size_cyto, offset_cyto, global_ther, clean,start_kernel = 2,end_karnel = 50,kernel_size = 20,outputTableName = None,outPath = None):
        self.start_kernel = start_kernel
        self.end_karnel = end_karnel
        self.kernel_size = kernel_size
        self.outputTableName = outputTableName
        self.outPath = outPath
        super().__init__(Image_name, block_size_cyto, offset_cyto, global_ther, clean)
        self.granularityMesure_globalSegmant()
    
    def openingOperation(self,kernel,image):
        '''
        Parameters
        ---------- 
            kernel: int, 
                size of filter kernel
        return
        ------
            opening operation image
        '''
        selem = morphology.disk(kernel, dtype=bool)
        eros_pix = morphology.erosion(image, footprint=selem)
        imageOpen = morphology.dilation(eros_pix, footprint=selem)
        return imageOpen
    
    
    def loopLabelimage(self,deploy = False ):
        '''
        return: dataframe of image intensity per segmented cell per image
        '''
        # returns column wise table
        open_vec = np.linspace(self.start_kernel, self.end_karnel, self.kernel_size, endpoint=True, dtype=int)
        if deploy:
            limit = 1
            open_vec = [open_vec[0],open_vec[-1]]
        else:
            limit = 3
        if len(open_vec) < limit:
            mesg = "increase kernel size"
            raise ValueError(mesg)
        mean_int = [] # image mean signal intensity
        sd_int = [] # image mean signal intensity
        for opning in open_vec:
            openImage = self.openingOperation(kernel = opning,image = self.ch2)
            mean_int.append(np.mean(openImage[self.mask>0]))
            sd_int.append(np.std(openImage[self.mask>0]))
        mean_int_norm = [int/mean_int[0] for int in mean_int]
        table = pd.DataFrame({'kernel':open_vec,'mean_decay':mean_int_norm,'mean':mean_int,'sd':sd_int })
        return table
    
    def granularityMesure_globalSegmant(self,saveMode = True):
        '''
        function description:
            - [] cell segmented using global threshold 
            - [] granularity measure
        output:
            Segmented image composition with area label
            - [] Area histogram plot
            - [] Segmented image composition after removal of small objects
            - [] plot of Mean intensity over opening operation (granularity spectrum)
        Parameters
        ----------
        clean: int
            remove object bellow the selected area size
        classLabel: int
            assign label
        file: str
        path: str
        outPath: str
        outputTableName: str
            e.g. "outputTableNorm.csv"
        saveMode: bool
            save returns
        Notes
        -----
        required single channel tif
        '''
        img = self.ch2 # from AIPS_Cyto_Global
        if len(img.shape) > 2:
            raise ValueError("required single channel tif")
        mask = self.mask 
        compImage = self.rgb_input_img #pil image
        granData = self.loopLabelimage()
        if saveMode:
            skimage.io.imsave(os.path.join(self.outPath,self.outputTableName + ".png"),compImage)
            granData.to_csv(os.path.join(self.outPath, self.outputTableName + ".png"))
        fig, ax = plt.subplots()
        sns.lineplot(data=granData, x="kernel", y="mean_decay").set(title='Granularity spectrum plot')
        

            

    



        