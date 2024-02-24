from AIPyS_old.segmentation.parametric.GlobalSeg import GlobalSeg
import numpy as np

class RunningWindow(GlobalSeg):
    '''
    Split the image according to the window size to identify the smallest image size with the highest possibility of identifying the phenotypic cell. Expect single image
    '''
    def __init__(self,windowSize,rnadomWindows,*args, **kwargs):
        self.windowSize = windowSize # for example split image to 6 slides
        self.rnadomWindows = rnadomWindows # for example add 3 more random slides
        super().__init__(*args, **kwargs)
        
    def runningWin(self,image_input,mask):
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
        maskSlice = []
        for i in range(0,minAxsis,stepSize):
            for j in range(0,minAxsis,stepSize):
                imageSlice.append(image_input[i:i+stepSize,j:j+stepSize])
                maskSlice.append(mask[i:i+stepSize,j:j+stepSize])
        # adding rando slices
        rnadomWindows = [1]*self.rnadomWindows
        range_xy = 0+stepSize,minAxsis-250
        while rnadomWindows:
            x = np.random.randint(low = range_xy[0],high = range_xy[1],size =1).tolist()[0]
            imageSlice.append(image_input[x:x+stepSize,x:x+stepSize])    
            maskSlice.append(mask[x:x+stepSize,x:x+stepSize])
            rnadomWindows.pop()
        return imageSlice, maskSlice
