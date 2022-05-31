'''
Subset tiles into training, testing, validation datasets
'''

# packages
import os
import sys
import shutil
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.io import imsave, imread, imshow

#dates of Planetscope satellite imagery

dates = ['May192020',
         'May152020',
         'June012020',
         'June032020',
         'June052020',
         'June142020',
         'June162020',
         'June272020',
         'July102020',
         'July122020',
         'July182020',
         'July242020']

for j in range(len(dates)): #for each date
    
	date = dates[j]           
    tile_size = 128
    valid_frac=0.2
    test_frac=0.2
	
	#create directories for training, testing, validation tiles
    path = f'F:/.../data/processed/training_data/{date}_2_data/' #Edit file path for your directory
    path_to_all_tiles = os.path.join(path,f'{date}_all_tiles_({tile_size}x{tile_size})')
    path_to_train_tiles = os.path.join(path,f'{date}_train_tiles_({tile_size}x{tile_size})')
    path_to_validation_tiles = os.path.join(path,f'{date}_valid_tiles_({tile_size}x{tile_size})')
    path_to_test_tiles = os.path.join(path,f'{date}_test_tiles_({tile_size}x{tile_size})')
    paths = [path_to_all_tiles, path_to_train_tiles, path_to_validation_tiles, path_to_test_tiles]
    
	for path_to_dir in paths:
        if not os.path.isdir(path_to_dir):
            os.mkdir(path_to_dir)
    files = os.listdir(path_to_all_tiles)
    
	split_index_1 = int(len(files)*valid_frac) #split tiles into validation fraction
    split_index_2 = int(len(files)*(1-test_frac)) #split tiles into testing fraction
    
    random.seed(1) #set random seed to randomly split tiles
    random.shuffle(files) #randomize tiles
	
    # validation split
    for file in files[:split_index_1]:
        file_out = path_to_all_tiles + '/' + file
        if os.path.isfile(file_out):
            shutil.copy2(file_out, path_to_validation_tiles)
			
    # train split
    for file in files[split_index_1:split_index_2]:
        file_out = path_to_all_tiles + '/' + file
        if os.path.isfile(file_out):
            shutil.copy2(file_out, path_to_train_tiles)
			
    # test split
    for file in files[split_index_2:]:
        file_out = path_to_all_tiles + '/' + file
        if os.path.isfile(file_out):
            shutil.copy2(file_out, path_to_test_tiles)
			
