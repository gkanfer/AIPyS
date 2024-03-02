deploying_aipys
===============

## Deploying AIPyS for Integration with Nikon-nis Elements

Leverage the AIPyS platform for efficient image processing, segmentation, analysis, and deployment file generation compatible with the Nikon-nis elements jobs module. Utilize the following command-line interface (CLI) instructions for a comprehensive workflow:

### Initial Image Selection and Parameter Adjustment

1. **Image Selection and Path Configuration:**

   ```
   updateParameters --Image_name "\dest\20X\WT\3_XY01.tif"
   ```

2. **Diameter Measurement via measDia (Web Application):**

   In this step, use the web application to measure the diameter across several sample cells. This helps establish analysis parameters:

   ```
   aipys --option measDia
   ```

3. **Update Estimated Cell Diameter:**

   Assuming a standard object diameter of 60 for a 20x objective:
   
   ```
   updateParameters --diameter 60
   ```

### Segmented Image Video Production

4. **Configure Video Generation Parameters:**

   Specify the details for video creation including, image count, video name, and related configurations:
   
   ```
   updateParameters --videoName "ImageSeqcp.avi" --data_dir "\dest\20X\PHENO" --imagesN 5 --outPath "\dest\AIPyS_output_images" --model_type cyto --channels greyscale
   ```

5. **Generate the Segmented Image Sequence with Cellpose:**

   ```
   aipys --option cp_seg_video
   ```

### Granularity Analysis

6. **Define Granularity Analysis Parameters:**

   ```
   updateParameters --videoName "GranMeasVideo_cp.avi" --start_kernel 2 --end_karnel 50 --kernel_size 20 --extract_pixel 50 --resize_pixel 150 --outputImageSize 500
   ```

7. **Produce Granularity Images:**

   ```
   aipys --option cp_gran_video
   ```

### Data Labeling and Visualization

8. **Save Single Cell Images & Compile an Intensity Table:**

   Specify the desired kernel size and set the base directory for gathering training data:

   ```
   updateParameters --kernelGran 6 --trainingDataPath "\dest\20X" --imagesN 5
   ```
   
   ```
   aipys --option cp_gran_table_gen
   ```

9. **Binary Labeling via dataLabeling (Web Application):**

   With this tool, users can perform binary phenotype labeling for the training dataset, improving the model's accuracy:

   ```
   updateParameters --imagePath "\dest\AIPyS_output_images\imageSequence\images" --dataPath "\dest\AIPyS_output_images\imageSequence\data"
   ```
   
   ```
   aipys --option dataLabeling
   ```

10. **Analyze Image Distribution with data_viz (Web Application):**

    Engage in data visualization to scrutinize image distribution and evaluate analytical results:

    ```
    updateParameters --imagePath "\dest\AIPyS_output_images\table_example\images" --dataPath "\dest\AIPyS_output_images\table_example"
    ```
    
    ```
    aipys --option data_viz
    ```

### Model Construction and Deployment File Preparation

11. **Build the Deployment Model:**

    Establish criteria for data training and model development:

    ```
    updateParameters --dataPath "\dest\AIPyS_output_images\table_example" --outPath "\dest\AIPyS_output_images" --imW 10 --imH 10 --thold 0.7 --areaSel 1000 --fractionData 50
    ```
    
    ```
    aipys --option modelBuild
    ```

12. **Generate Deployment Build File:**

    After the model is finalized, ready the deployment file for integration with Nikon-nis elements:

    ```
    updateParameters --Image_name "\dest\022224\2.tif" --outPath "\dest\AIPyS_output_images\outproc_temp"
    ```
    
    ```
    aipys --option deployBuild
    ```

### Parameter Management

For adjustments or resets of parameters at any phase:

- **For Help:** `load-parameters --help`
- **To Generate Default Parameters:** `load-parameters --select generate`
- **To Reset Parameters:** `load-parameters --select reset`
- **To Display Current Settings:** `load-parameters --select display`

Ensure to align paths to your directory configurations. The full suite of CLI instructions provides a detailed process from initiating the AIPyS project to model readiness and subsequent deployment, with enhanced focus on web application functions for user engagement and analytic insights.