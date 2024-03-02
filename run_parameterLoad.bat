@echo on
call activate C:\Users\youle_lab\anaconda3\envs\cellpose_pymc_dash
call python .\AIPyS\CLI\set_parameters.py --Image_name D:\Gil\images\pex_project\20X\WT\3_XY01.tif
call python .\AIPyS\CLI\set_parameters.py --diameter 60
call python .\AIPyS\CLI\set_parameters.py --videoName ImageSeqcp.avi --data_dir D:\Gil\images\pex_project\20X\PEX3 --imagesN 5 --outPath D:\Gil\images\pex_project\AIPyS_output_images --videoName ImageSeqcp.avi --model_type cyto --channels greyscale
call python .\AIPyS\CLI\set_parameters.py --videoName GranMeasVideo_cp.avi  --start_kernel 2 --end_karnel 50 --kernel_size 20 --extract_pixel 50 --resize_pixel 150 --outputImageSize 500
call python .\AIPyS\CLI\set_parameters.py --kernelGran 6 --trainingDataPath D:\Gil\images\pex_project\20X --imagesN 5
call python .\AIPyS\CLI\set_parameters.py --imagesN 2 --outPath D:\Gil\images\pex_project\AIPyS_output_images\pexanalysis
call python .\AIPyS\CLI\set_parameters.py --dataPath D:\Gil\images\pex_project\AIPyS_output_images\table_example --outPath D:\Gil\images\pex_project\AIPyS_output_images --imW 10 --imH 10 --thold 0.7 --areaSel 1000