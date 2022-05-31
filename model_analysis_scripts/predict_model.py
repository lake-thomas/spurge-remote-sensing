'''
Function to predict on satellite imagery after training deep learning model
Inputs include a satellite image, model weights  and model file
Output prediction mosaic for the input satellite image

'''

#Import Library
import os
import argparse
import gdal
import numpy as np
from tqdm import tqdm
from IPython.display import HTML, display
from tabulate import tabulate


import tensorflow as tf
import keras
import keras.backend as K
from tensorflow.keras.models import load_model
#import segmentation_models as sm
#from segmentation_models.losses import CategoricalCELoss

# Land Cover Classes

number_of_classes = 7
	
color_map = [
    [0, 0, 0], #0 background, black
    [186,236,104], #1 grass/shrubs, light green
    [238,14,10], #2 buildings, brown
    [113, 113, 113], #3 impervious, dark grey
    [67,110,238], #5 lakesponds, dark blue
    [13, 77, 12], #6 ag, darker green
    [0,0,0]] #7 leafy spurge, red

# Save GeoTIFF Image

def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    height, width = image_data.shape
    dataset       = driver.Create(path, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data)
    color_table = gdal.ColorTable()
    for i in range(number_of_classes):
        color_table.SetColorEntry(i, tuple(color_map[i]) + (255,))
    dataset.GetRasterBand(1).SetRasterColorTable(color_table)
    dataset.GetRasterBand(1).SetNoDataValue(255)
    dataset.FlushCache()

    
def eval_image(input_image_path, model, output_image_path):
    #input_image_path = r'F:\...\test prediction tiles\images\20200612_164542_101f_3B_AnalyticMS_SR.tif' #change for your file path
    classes = ['background', 'vegetation', 'buildings', 'roads', 'water', 'ag', 'leafyspurge']
    input_dataset = gdal.Open(input_image_path)
    input_image   = input_dataset.ReadAsArray()
    print(input_image.shape)
    input_image = np.rollaxis(input_image, 0, 3)
    h, w, n     = input_image.shape
    #input_image[:, :, :n_features] = preprocessing_image_ms_subset(input_image[:, :, :n_features])
    print(input_image.shape)
    input_image = input_image[:, :, 0:4]
    #standardize
    #mean, std = input_image.mean(), input_image.std()
    #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    #input_image = (input_image - mean) / std
    # normalize
    #input_image = 2*(input_image - input_image.min())/(input_image.max()-input_image.min())-1
    model_input_height = 128
    model_input_width = 128
    model_input_channels = 4
    print(model_input_height, model_input_width, model_input_channels)
    model_output_height = 128
    model_output_width = 128
    model_output_channels = 7
    #model.layers[len(model.layers) - 1].output_shape[1:4]
    print(model_output_height, model_output_width, model_output_channels)
    padding_y = int((model_input_height - model_output_height)/2)
    padding_x = int((model_input_width - model_output_width)/2)
    assert model_output_channels == number_of_classes
    pred_lc_image = np.zeros((h, w, number_of_classes))
    print(pred_lc_image.shape)
    mask = np.ones((h, w))
    print(mask.shape)
    irows, icols = [],[]
    batch_size   = 1
    minibatch    = []
    ibatch       = 0
    mb_array     = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
    print(mb_array.shape)
    n_rows = int(h / model_output_height)
    print(n_rows)
    n_cols = int(w / model_output_width)
    print(n_cols)
    for row_idx in tqdm(range(n_rows)):
        for col_idx in range(n_cols):
            subimage = input_image[row_idx*model_output_height:row_idx*model_output_height + model_input_height, col_idx*model_output_width:col_idx*model_output_width + model_input_width, :]
            mb_array[ibatch] = subimage
            ibatch += 1
            irows.append((row_idx*model_output_height + padding_y,row_idx*model_output_height + model_input_height - padding_y))
            icols.append((col_idx*model_output_width +  padding_x,col_idx*model_output_width  + model_input_width  - padding_x))
            if (ibatch) == batch_size:
                outputs = model.predict(mb_array)
                for i in range(batch_size):
                    r0,r1 = irows[i]
                    c0,c1 = icols[i]
                    pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
                    mask[r0:r1, c0:c1] = 0
                ibatch = 0
                irows,icols = [],[]
    label_image = np.ma.array(pred_lc_image.argmax(axis=-1), mask = mask)
    save_image(label_image.filled(255), output_image_path, input_dataset.GetGeoTransform(), input_dataset.GetProjection())
    #save_image(pred_lc_image[:, :, 6], output_image_path, input_dataset.GetGeoTransform(), input_dataset.GetProjection()) #use to get softmax predictions for a given class

	
# Evaluate Images in Folder

def evaluate(input_dir, model_path, output_dir):
    model = load_model(model_path)
    for items in os.listdir(os.path.join(input_dir, 'images')):
        if items.endswith(".tif"):
            pth = os.path.join(os.path.join(input_dir, 'images'), items)
            out_pth = pth.replace('images', 'results')
            eval_image(pth, model, out_pth)
            print('saved result to ' + out_pth)
			
			
# Set Parameters

#input directory should hold two folders, one with 'images' and other with empty folder 'results'
input_dir = r'E:\test prediction tiles'

#model path should be path to model results .h5 file (not model weights or json file) from the model.save function
model_path =r'F:\your_trained_model.h5'

##both model and model weights must be loaded in order to predict model
output_image_dir = r'E:\test prediction tiles\results'

# Run Script Keras
evaluate(input_dir, model_path, output_image_dir)
