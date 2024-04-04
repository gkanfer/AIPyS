from AIPyS.segmentation.cellpose.AIPS_cellpse_video_gen import CellPoseSeg
import pandas as pd
import glob
def main():
    path = 'D:\\Gil\\images\\pex_project\\10X\\Pex3\\nd_001'
    outpath = "D:\\Gil\\images\\pex_project\\10X"
    files = glob.glob(path + "\\*.tif")
    CellPoseSeg(diameter =  60, videoName = "test_10x.avi", model_type = 'cyto', channels = [0,0],
                    Image_name = files[:3], outPath = outpath)
    print(df)
if __name__ == "__main__":
    main()