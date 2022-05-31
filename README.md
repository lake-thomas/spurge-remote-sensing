# Remote Sensing of Invasive Species Leafy Spurge with Convolutional Neural Networks and Long-Short-Term Memory Networks

This repo contains code for training and testing the neural networks described in this paper: *Lake, Thomas; Briscoe Runquist, Ryan; & Moeller, David. (2022). Deep learning detects invasive plant species across complex landscapes using Worldview-2 and Planetscope satellite imagery. Remote Sensing in Ecology and Conservation, Vol XX DOI XXX.

We developed all models in Python v3.6.9 with Keras v2.2.4 (Chollet et al., 2015) and TensorFlow v2.0.0 (Abadi et al., 2016). We made special use of the segmentation-models package (https://github.com/qubvel/segmentation_models) to access and train the SE-ResNet50 U-net architecture (Yakubovskiy, 2019).


![image](https://media.github.umn.edu/user/12668/files/8e4aa68a-76d3-4e82-acce-32af97bf0142)


## Setup
----------------------
Use of a conda virtual environment is encouraged when managing different package versions in Python. The exact package versions used for the analyses are included in the spurge-remotesensing-condaenv.txt file. Note: not all packages in the text file are required for analyses, most imporant dependencies are Keras, Tensorflow, Numpy, Gdal libraries.

$ conda env create --file spurge-remotesensing-condaenv.txt

## Files

Included in *data/* folder are several sample satellite image tiles from the Planetscope Dove and Worldview-2 imagery that show the file structure and model inputs used for training. The large file size of satellite images and limited redistribution permissions restricts the upload of the full dataset used for analyses, but satellite image metadata are available in the manuscript supplementary methods. 

The *data_prep_scripts/* folder contains several short scripts to aid in processing satellite images and a ground truth raster, as well as dividing images randomly into training, testing, and validation sets. 

----------------------
- data/
  - planetscope_cnn/
    - Sample satellite image tiles of Planetscope Dove imagery for CNN
  - planetscope_lstm/
    - Sample satellite image tiles of Planetscope Dove imagery for LSTM 
  - worldview_cnn/
    - Sample satellite image tiles of Worldview-2 imagery for CNN
- data_prep_scripts/
  * merge_rasters_gdal: Gdal_merge.py script to combine a GeoTiff with a ground truth raster
  * split_geotiff_into_tiles: Python script to split a GeoTiff file into smaller tiles for training with a CNN
  * split_geotiff_into_training_testing_valid: Python script to subset tiles randomly into training, testing, validation datasets
* model_analysis_scripts/
  * confusion_matrix_plots: Python script to plot confusion matrix with and without normalization from ground truth and predicted datasets
  * model_summary_statistics: Python script to calculate TP, TN, FP, FN and model summary statistics from model predictions
  * plot_intermediate_activation_layers: Python script to view intermaediate activation layers and softmax outputs from model predictions
  * predict_model: Python script to generate model predictions for image tiles and mosaic predictions, output as a .tif file
  * LSTM_temporal_occlusion_script: Python script selects and applies random noise to a time-series of satellite imagery, useful to ask which time periods are important for the LSTM to predict land cover classes
  * CNN_spectral_occlusion_script: Python script selects and applies spectral noise sampled from the empirical distribution of leafy spurge and vegetation spectral reflectances to satellite imagery, useful to ask which spectral bands are important for the CNN to predict land cover classes.
* models/
  * CNN_Planetscope: Python script to train and validate CNN model using Planetscope Dove satellite imagery
  * CNN_Worldview2: Python script to train and validate CNN model using Worldview-2 satellite imagery
  * CNN_LSTM_Planetscope: Python script to train and validate CNN-LSTM model using Planetscope Dove satellite imagery

## Abstract
----------------------
Effective management of invasive species requires rapid detection and dynamic monitoring. Remote sensing offers an efficient alternative to field surveys for invasive plants; however, distinguishing individual plant species can be challenging especially over geographic scales. Satellite imagery is the most practical source of data for developing predictive models over landscapes, but spatial resolution and spectral information can be limiting. We applied neural networks to two types of satellite imagery to detect the invasive plant, leafy spurge (Euphorbia virgata), across a heterogeneous landscape in Minnesota, USA. We developed convolutional neural networks (CNNs) with imagery from Worldview-2 and Planetscope satellites. Worldview-2 imagery has high spatial and spectral resolution, but images are not routinely taken in space or time. By contrast, Planetscope imagery has lower spatial and spectral resolution, but images are taken daily across Earth. The former had 96.1% accuracy in detecting leafy spurge whereas the latter had 89.9% accuracy. Second, we modified the CNN for Planetscope with a long short-term memory (LSTM) layer that leverages information on phenology from a time series of images. The detection accuracy of the Planetscope LSTM model was 96.3%, on par with the high resolution, Worldview-2 model. Across models, most false positive errors occurred near true populations, indicating that these errors are not consequential for management. We identified that early and mid-season phenological periods in the Planetscope time series were key to predicting leafy spurge. Additionally, green, red-edge, and near-infrared spectral bands were important for differentiating leafy spurge from other vegetation. These findings suggest that deep learning models can accurately identify individual species over complex landscapes even with satellite imagery of modest spatial and spectral resolution if a temporal series of images is incorporated. Our results will help inform future management efforts using remote sensing to identify invasive plants, especially across large-scale, remote, and data-sparse areas.

## Associated Publication
----------------------
Lake, T.A., Briscoe Runquist, R.D., Moeller, D.A. 2022. Deep learning detects invasive plant species across complex landscapes using Worldview-2 and Planetscope satellite imagery. _Remote Sensing in Ecology and Conservation_ XX(X), XXX; doi: XXXX

## Notice
----------------------
Funding for this project was provided by the Minnesota Invasive Terrestrial Plants and Pests Center through the Environment and Natural Resources Trust Fund as recommended by the Legislative-Citizen Commission on Minnesota Resources (LCCMR). We thank the Maxar Technologies and the European Space Agency for providing access to Worldview-2 satellite imagery (Proposal Id: 63324). We gratefully acknowledge the support of NVIDIA Corporation for the donation of the Titan V GPU used for this research. 
