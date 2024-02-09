from AIPyS.classification.bayes.RunningWindow import RunningWindow
import numpy as np
import string

class GranularityDataGen(RunningWindow):
    def __init__(self,*args, **kwargs):
        self.length = length # lenagth of the random name for the image:e.g. 7
        super().__init__(*args, **kwargs)

    def generate_random_string(self,length):
        letters = string.ascii_letters  # this includes both lowercase and uppercase letters
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str 

    def imageMeasure(inage_input, mask,IntenTimeZero):
        return np.mean(inage_input[mask>0])/IntenTimeZero, np.mean(inage_input[mask>0]), np.std(inage_input[mask>0])
      
    def getMeanSD(self,input_image):
        return np.round(np.mean(input_image),3),np.round(np.std(input_image),3)

    def granCalc(self):
        image_name = self.generate_random_string(7)
        tempFolder = os.path.join(self.outPath,image_name)
        ImageOrig = os.path.join(tempFolder,'ImageOrig')
        ImageSplit = os.path.join(tempFolder,'ImageSplit')
        os.makedirs(tempFolder,exist_ok=True)
        os.makedirs(ImageOrig,exist_ok=True)
        os.makedirs(ImageSplit,exist_ok=True)
        image = self.img # should be single image
        IntenTimeZero = np.mean(
        mask = self.cytosolSegmentation(image)
        IntenTimeZero = np.mean(image[mask>0])
        imageSlice,maskSlice = self.runningWin(image_input = image, mask = mask)
        dfGran = {'name':[],'intensityRatio':[],'intensity':[],'sd':[]}
        name, intensityRatio, intensity, sd = [],[],[],[]
        for i,imageCrop,maskCrop in enumerate(imageSlice,maskSlice):
            imageDecy = self.openingOperation(kernel,image = imageCrop)
            ratio,mean,sd = self.imageMeasure(imageDecy, maskCrop)
            dfGran['name'].append(image_name)
            dfGran['ratio'].append(ratio)
            dfGran['intensity'].append(mean)
            dfGran['sd'].append(sd)
            skimage.io.imsave(os.path.join(ImageSplit,image_name + f'_mean_{mean}' + f'_sd_{sd}_' +'.tif'), imageCrop)
            print('-' * i, end = '\r')



