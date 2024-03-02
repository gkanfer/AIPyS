# Bayes Classifier Deployment
The Baye's Granularity model created is exported and then utilized for deployment on the Nikon NIS Elements HCT package. A bash file was employed to direct the NIS jobs module to utilize AIPyS. The AIPys granularity classifier requires several parameters to assess the granularity resulting from the Baye's Granularity training. 
```bash
@echo on
call activate AIPys_conda_env
call python D:\run_Bayes_Classifier.py
@pause
```
When obtaining an image, a single-channel image is taken and then stored on the local system.


```python
from AIPyS.Baysian_deploy import BayesianGranularityDeploy
file = 'input.tif'
path_input = r'C:\NIS\outproc'
path_out = path_input

BayesianGranularityDeploy(file = file, path = path_input, kernel_size = 15, trace_a = -27, trace_b = 33, 
                          thold = 0.7,   pathOut = path_out,clean = 500,saveMerge=True)
```
The BayesianGranularityDeploy function returns a binary mask of the cells that represent the chosen phenotype. This mask is saved as ```binary.tif``` and then uploaded to the NIS-Elements module, where it is converted into a Region of Interest (ROI). The simulation module then takes the photostimulation raster and uses a UV laser to activate those regions.


