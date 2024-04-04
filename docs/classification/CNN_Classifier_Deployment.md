# CNN_Classifier_Deployment
Instead of using Bayes Granularity to deploy a model, the AIPyS Convolutional Neural Network (CNN) model can be used for detecting the desired phenotypes. To use the CNN model, it must be exported and then utilized with the Nikon NIS Elements HCT package. A bash file is used to instruct the NIS jobs module to run the AIPyS.

```bash
@echo on
call activate AIPys_conda_env
call python D:\run_CNN_Classifier.py
@pause
```

When obtaining an image, a single-channel image is taken and then stored on the local system.

```python
from AIPyS.Baysian_deploy import BayesianGranularityDeploy
file_name = 'input.tif'
path_model = 'data'
path_input = 'C:\NIS\outproc'
path_out = path_input

CNNDeploy(path_model = path_model, model = 'cnn.h5',
          file_name = file_name, path = path_input, pathOut = path_out,
          areaFilter = 1500, thr=0.5)
```

The CNNDeploy function returns a binary mask of the cells that represent the chosen phenotype. This mask is saved as binary.tif and then uploaded to the NIS-Elements module, where it is converted into a Region of Interest (ROI). The simulation module then takes the photostimulation raster and uses a UV laser to activate those regions.



```python

```
