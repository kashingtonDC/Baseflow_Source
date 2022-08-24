import os
import fiona
import datetime
import numpy as np
import pandas as pd
import geopandas as gp
import rasterio as rio

from tqdm import tqdm
from rasterio.mask import mask


def read_and_mask(fn,area_geom,scale_factor):
	
	src = rio.open(fn) # Read file
	src2 = rio.mask.mask(src, area_geom, crop=True) # Clip to shp 

	fl_arr = src2[0].astype(np.float) # read as array
	arr = fl_arr.reshape(fl_arr.shape[1], fl_arr.shape[2]) / scale_factor # Reshape bc rasterio has a different dim ordering 
	outarr = np.where(arr < 0 ,np.nan, arr)# Mask nodata vals 
	arr = arr/scale_factor # divide by scale factor

	return outarr

def main():
	
	# Read GDF
	gdf = gp.read_file("../shape/sierra_catchments.shp")

	# Setup dirs 
	et_dir = "../data/ssebop/processed"

	# Get lists of files 
	et_files = [os.path.join(et_dir,x) for x in os.listdir(et_dir) if x.endswith(".tif")]

	# Sort
	et_files.sort()

	# Datetime the start/end
	start = datetime.datetime.strptime("2001-01-01", "%Y-%m-%d")
	end = datetime.datetime.strptime("2021-12-31", "%Y-%m-%d")
	dt_idx = pd.date_range(start,end, freq='D')

	# Datetime objs --> strs
	doys = [str(x.timetuple().tm_yday) for x in dt_idx]
	doys_3d = []
	for doy in doys:
		if len(doy) == 2:
			doys_3d.append("0" + doy)
		else:
			doys_3d.append(doy)
	
	d1strs = [y.strftime('%Y') + doy for y, doy in zip(dt_idx,doys_3d)]

	print(d1strs)

	# Read catchments 
	gdf = gp.read_file("../shape/sierra_catchments.shp")
	stn_lookup = dict(zip(gdf['stid'], [x[:3] for x in gdf['catch_name']]))

	# Loop through watersheds
	stn_id_list = [x for x in list(gdf['stid'])]

	print(stn_id_list)

	for stn_id in stn_id_list[:]:

		print("PROCESSING {}  ======================".format(stn_id))

		# Read shapefile for mask
		shppath = "../shape/{}.shp".format(stn_id)

		with fiona.open(shppath, "r") as shapefile:
			area_geom = [feature["geometry"] for feature in shapefile]
		
		et_arrs = {}

		# Make a blank array for the dates that don't exist 
		ref_src = rio.open(et_files[0])
		ref_src_masked = mask(ref_src, area_geom, crop=True)[0] # Clip to shp 
		ref_arr_masked = ref_src_masked.reshape(ref_src_masked.shape[1], ref_src_masked.shape[2]) 
		ref_arr_0 = np.zeros_like(ref_arr_masked)
		ref_arr = np.where(ref_arr_0==0,np.nan,ref_arr_0)

		# Loop through each day 
		for datestr in tqdm(d1strs[:]):

			et_fn = [x for x in et_files if datestr in x]

			 # Add nan ims for the dates with no data (listed in SI)
			if len(et_fn) == 0:
				print("ET {} MISSING".format(datestr))
				et_arr = ref_arr.copy()
			else:
				et_arr = read_and_mask(et_fn[0],area_geom,scale_factor= 1000)

			et_arrs[datestr] = et_arr

		et_out = np.dstack(list(et_arrs.values()))

		if not os.path.exists("../data/Watersheds"):
			os.mkdir("../data/Watersheds")

		np.save("../data/Watersheds/{}_et.npy".format(stn_id), et_out )

		print("********" *5)
		print("WROTE FILES FOR {} in ../data/Watersheds ".format(stn_id))
		print("********" *5)


if __name__ == '__main__':
	main()
			
