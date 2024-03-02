from AIPyS.Baysian_deploy import BayesianGranularityDeploy
file = 'input.tif'
path_input = r'C:\NIS\outproc'
path_out = path_input
BayesianGranularityDeploy(file =file, path = path_input, kernel_size = 14, trace_a = -9, trace_b = 14.5, thold = 0.7,   pathOut = path_out, clean = 500,saveMerge=True)