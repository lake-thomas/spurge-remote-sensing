"""
This script splits a Geotiff file into smaller tiles for training with a CNN.
"""

# packages

import os
import sys
import shutil
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.io import imsave, imread, imshow

#define geotiff and tile size parameters

date = "July102020"
aoi = "AOI0"
tile_size = 128
is_training_data = True
tile_separation = 1

# construct paths
filename = f'{date}_ps_{aoi}_merged_image_gt.tif'
dirname = f'{date}_all_tiles_({tile_size}x{tile_size})'

path = f'F:/.../data/processed/training_data/{aoi}/{date}_2_data/'

path_to_all_tiles = os.path.join(path, dirname)

if not os.path.isdir(path_to_all_tiles):
    os.makedirs(path_to_all_tiles)

# split into tiles

print('\nsplitting image into tiles...', end='')
with rasterio.open(path+filename) as tif:
    height = tif.profile['height']
    width = tif.profile['width']
    if is_training_data:
        nr_tile_rows = height // (tile_size + tile_separation)
        nr_tile_cols = width // (tile_size + tile_separation)
        for i in range(nr_tile_rows):
            for j in range(nr_tile_cols):
                # get a tile
                # Window syntax:
                # Window(col_off, row_off, width, height)
                tile = tif.read(window=Window(j*(tile_size+tile_separation),
                                              i*(tile_size+tile_separation),
                                              tile_size,
                                              tile_size))
                # get from shape=(channels, height, width)
                # to (height, width, channels)
                tile = np.swapaxes(tile, 0, 2)
                tile = np.swapaxes(tile, 0, 1)
                imsave(os.path.join(path,dirname,f'{aoi}_{date}_tile_{i:03d}_{j:03d}.tif'),tile)
