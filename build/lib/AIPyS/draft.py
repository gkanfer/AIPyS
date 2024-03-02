from AIPyS.classification.bayes.GranularityDataGen import GranularityDataGen_cp
Image_name = r'D:\Gil\images\pex_project\20X\PEX3\1_XY03.tif' 
outPath = r'D:\Gil\temp'
model_type = 'cyto'
channels = [0,0] 
diameter = 50
start_kernel

GranularityDataGen_cp(kernelGran = 6,w = 500,h = 500,
                    extract_pixel = 50, resize_pixel = 150,
                    model_type = model_type, channels = channels,
                    Image_name = Image_name, outPath = outPath)
