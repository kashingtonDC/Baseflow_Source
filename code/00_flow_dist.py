
import os
import time
import fiona
import datetime
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import rasterio as rio
import geopandas as gp

from tqdm import tqdm
from pysheds.grid import Grid

# # Delineate watersheds of the Sierra Nevada
# * Use this knowtebook to demo DEM + flow routing algorithms + reservoir locations to delineate upstream catchments; key for reservoir operations, agricultural, and domestic water supply in California
# 
# ### Get the elevation data and preproces
# * Use Digital Elevation Models (DEMs) and flow routing algorithms
# * Shuttle radar topography mission (SRTM) DEMS downloaded from: http://srtm.csi.cgiar.org/srtmdata/
# * Merged by running from command line: 
# 
# `gdal_merge.py -o srtm_dem.tif srtm_12_04.tif srtm_12_05.tif srtm_13_04.tif srtm_13_05.tif srtm_13_06.tif`
#     
# * Clipped by running from command line:
# 
# `gdalwarp -cutline ../shape/cvws.shp -crop_to_cutline srtm_dem.tif hu6_srtm_dem.tif`

# ### Implement flow routing algorithms
# * Use Pysheds package: https://github.com/mdbartos/pysheds (Thanks to @mdbartos for a very nifty package!) 
# * (1) DEM --> (2) Flow Direction --> (3) Flow Accumulation --> (4) Adjust Pour Points --> (5) Watershed Delineation --> (6) Flow distances from outlet 


# Read the file we wrote and edited 
outflows = gp.read_file("../shape/trm_res_fnf.shp")
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

# Calc the distance from outlet of each pixel 
# TODO: Parallelize and run over a weekend - This takes FOREVER to run

for i in tqdm(outflows[:].iterrows()):

    # Select outlet from outflows df
    df = outflows.iloc[i[0]]
    x,y = df.geometry.x, df.geometry.y
    
    stid = df['STA']
    
    # Reload the grid because we write in place each time
    grid = Grid.from_raster('../rasters/flowdir_clip.tif', data_name='dir')
    # Select outlet from outflows df

        
    # comp flow distance 
    print("Processing "  + stid)    # Delineate the catchment
    grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
    
    grid.flow_distance(x, y, 'dir', xytype='label', dirmap=dirmap,
                   out_name='dist', nodata_out=np.nan, method='shortest')
    
    # save
    outdir = "../rasters/flow_dist"
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    grid.to_raster('dist', '../rasters/flow_dist/{}_dist.tif'.format(stid))
    
    npix = np.nansum(grid.view('catch') > 0)
    wshed_area = 90*90*int(npix)*1e-6
    
    print("wrote ../rasters/{}_dist.shp".format(stid))    
    print("=======" * 7)
    


