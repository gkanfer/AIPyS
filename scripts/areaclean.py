from AIPyS.classification.bayes.GranularityDataGen import GranularityDataGen_cp

import pandas as pd
import glob
def main():
    path = 'D:\\Gil\\images\\pex_project\\10X\\Pex3\\nd_001'
    # outpath = "D:\\Gil\\images\\pex_project\\10X"
    files = glob.glob(path + "\\*.tif")
    model_type = "cyto"
    outPath = 'D:\\Gil\\temp'
    trainingDataPath = "D:\\Gil\\images\\pex_project\\20X\\PEX3"
    GranularityDataGen_cp(kernelGran = 6,w = 500,h = 500,
                            extract_pixel = 50, resize_pixel = 150,diameter =  60,
                            model_type = model_type, channels = "greyscale",
                            Image_name = files[:3], outPath =outPath,trainingDataPath = trainingDataPath)
if __name__ == "__main__":
    main()