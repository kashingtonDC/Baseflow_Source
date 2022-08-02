import os

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gp
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from osgeo import gdal, osr
from scipy import stats, spatial, signal, fftpack

import warnings
warnings.filterwarnings("ignore")


# Functions

def write_raster(array,gdf,outfn):
	'''
	converts a numpy array and a geopandas gdf to a geotiff
	Data values are stored in np.array
	spatial coordinates stored in gdf
	outfn - outpath
	'''

	xmin, ymin = gdf.bounds.minx.values[0], gdf.bounds.miny.values[0]
	xmax, ymax = gdf.bounds.maxx.values[0], gdf.bounds.maxy.values[0]
	nrows, ncols = array.shape
	xres = (xmax-xmin)/float(ncols)
	yres = (ymax-ymin)/float(nrows)
	geotransform =(xmin,xres,0,ymax,0, -yres)   

	output_raster = gdal.GetDriverByName('GTiff').Create(outfn,ncols, nrows, 1 , gdal.GDT_Float32)  # Open the file
	output_raster.SetGeoTransform(geotransform)  # Specify coords
	srs = osr.SpatialReference()                 # Establish encoding
	srs.ImportFromEPSG(4326)                     # WGS84 lat long
	output_raster.SetProjection(srs.ExportToWkt() )   # Export coordinate system 
	output_raster.GetRasterBand(1).WriteArray(array)   # Write array to raster

	print("wrote {}".format(outfn))
	return outfn


def calc_nbins(N):

	'''
	A. Hacine-Gharbi, P. Ravier, "Low bias histogram-based estimation of mutual information for feature selection", Pattern Recognit. Lett (2012).
	'''
	ee = np.cbrt(8 + 324*N + 12*np.sqrt(36*N + 729*N**2))
	bins = np.round(ee/6 + 2/(3*ee) + 1/3)

	return int(bins)

def calc_mi(imstack, inflow):

	# Build the out image
	mi_im = np.zeros_like(np.mean(imstack, axis = 2))

	rows, cols, time = imstack.shape
	px_ts = []
	rclist = []

	# extract pixelwise timeseries
	for row in range(rows):
		for col in range(cols):
			ts_arr = imstack[row,col,:]

			if not np.isnan(ts_arr).all():
				px_ts.append(pd.Series(ts_arr))
				rclist.append([row,col])
			else:
				px_ts.append(pd.Series(np.zeros_like(ts_arr)))
				rclist.append([row,col])

	pxdf = pd.concat(px_ts, axis = 1)
	pxdf.columns = pxdf.columns.map(str)

	# Populate the per-pixel lags 
	for rc, dfcolidx in tqdm(list(zip(rclist,pxdf.columns))):

		tempdf = pd.DataFrame([pxdf[dfcolidx].copy(),inflow]).T
		tempdf.columns = ['var','q']

		# get n bins
		nbins = calc_nbins(len(tempdf))

		# compute mutual info
		try: 
			mi = metrics.mutual_info_score(tempdf['var'].value_counts(normalize=True,bins = nbins),tempdf['q'].value_counts(normalize=True,bins = nbins))
		except:
			mi = np.nan

		# fill ims
		rowidx, colidx = rc
		mi_im[rowidx,colidx] = mi

	return mi_im

def normalize(x):
	return(x-np.nanmin(x))/(np.nanmax(x)- np.nanmin(x))

def calc_xcorr_fft(imstack, qarr):
	rows, cols, time = imstack.shape
	px_ts = []
	rclist = []

	# extract pixelwise timeseries
	for row in range(rows):
		for col in range(cols):
			ts_arr = imstack[row,col,:]

			if not np.isnan(ts_arr).all():
				px_ts.append(pd.Series(ts_arr))
				rclist.append([row,col])
			else:
				px_ts.append(pd.Series(np.zeros_like(ts_arr)))
				rclist.append([row,col])

	pxdf = pd.concat(px_ts, axis = 1)
	pxdf.columns = pxdf.columns.map(str)

	# Build the out image
	lagim = np.zeros_like(np.mean(imstack, axis = 2))
	corrim = np.zeros_like(np.mean(imstack, axis = 2))
	pvalim = np.zeros_like(np.mean(imstack, axis = 2))

	# Populate the per-pixel lags 
	for rc, dfcolidx in tqdm(list(zip(rclist,pxdf.columns))):

		a=pxdf[dfcolidx].values
		b=qarr.copy()

		# compute shift + corr mag

		# Shift
		try:
			A = fftpack.fft(normalize(a))
			B = fftpack.fft(normalize(b))
			Ar = -A.conjugate()
			shiftval = np.argmax(np.abs(fftpack.ifft(Ar*B))[:]) # 365 day buffer for reasonable results 
		except:
			shiftval = np.nan

		try:
			corrcoef = stats.pearsonr(a,b)
			corr = corrcoef[0]
			pval = corrcoef[1]
		except:
			pval = np.nan
			corr = np.nan

		# fill ims
		rowidx, colidx = rc
		lagim[rowidx,colidx] = shiftval
		corrim[rowidx,colidx] = abs(corr)
		pvalim[rowidx,colidx] = pval

	return lagim.astype(float), corrim.astype(float), pvalim.astype(float)

def fft_wrapper(imstack,qarr):

	lag_im, corr_im, pval_im = calc_xcorr_fft(imstack,qarr)

	# Get mean of theentire stack 
	im_mean = np.nanmean(imstack, axis = 2)

	# Mask zeros
	# lag_im, corr_im = [np.where(x==0, np.nan, x) for x in [lag_im,corr_im]]

	# Filter lag and cor by >0.001 mm tottalthreshold for smlt and >1 for precip 
	lag_im, corr_im = [np.where(im_mean<np.nanpercentile(im_mean,5), np.nan, x) for x in [lag_im,corr_im]]

	# Filter lag and cor by P value >0.05
	lag_im, corr_im = [np.where(pval_im>0.05, np.nan, x) for x in [lag_im,corr_im]]

	lag_im = np.where(np.isnan(pval_im), np.nan, lag_im)

	return lag_im, corr_im

def mi_wrapper(imstack,qarr):

	mi_im = calc_mi(imstack,qarr)

	# Get mean of theentire stack 
	im_sum = np.nansum(imstack, axis = 2)

	# mask where sum is zero 
	mi_im = np.where(im_sum==0, np.nan, mi_im)

	# Mask the other nans 
	mi_im = np.where(np.isnan(im_sum), np.nan, mi_im)
	
	return mi_im


def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def calc_rolling_xcorr(imstack,qarr, win_size = 30):
	
	rows, cols, time = imstack.shape
	px_ts = []
	rclist = []

	# extract pixelwise timeseries
	for row in range(rows):
		for col in range(cols):
			ts_arr = imstack[row,col,:]

			if not np.isnan(ts_arr).all():
				px_ts.append(pd.Series(ts_arr))
				rclist.append([row,col])
			else:
				px_ts.append(pd.Series(np.zeros_like(ts_arr)))
				rclist.append([row,col])

	pxdf = pd.concat(px_ts, axis = 1)
	pxdf.columns = pxdf.columns.map(str)

	# set up out ims 
	lagim = np.zeros_like(np.mean(imstack, axis = 2))
	corrim = np.zeros_like(np.mean(imstack, axis = 2))

	# Loop through rows / cols
	for rc, dfcolidx in tqdm(list(zip(rclist,pxdf.columns))[:]):

		a=pxdf[dfcolidx].values
		b=qarr.copy()

		# Split into windows
		var_wins = rolling_window(a,win_size)
		q_wins = rolling_window(b,win_size)

		lags = []
		corrs = []
		pvals = []

		for vwin, qwin in list(zip(var_wins, q_wins))[:]:
			# Shift
			try:
				A = fftpack.fft(normalize(vwin))
				B = fftpack.fft(normalize(qwin))
				Ar = -A.conjugate()
				shiftval = np.argmax(np.abs(fftpack.ifft(Ar*B)))
			except:
				shiftval = np.nan

			try:
				corrcoef = stats.pearsonr(vwin,qwin)
				corr = corrcoef[0]
				pval = corrcoef[1]
			except:
				pval = np.nan
				corr = np.nan

			lags.append(shiftval)
			corrs.append(corr)
			pvals.append(pval)
	
		lcpdf = pd.DataFrame([np.array(lags), np.array(corrs), np.array(pvals)]).T

		lcpdf.columns = ['lag', 'corr', 'pval']
		lcpdf = lcpdf[lcpdf['pval']<0.05]

		corr = lcpdf['corr'].mean()
		shiftval = lcpdf['lag'].mean()
	
		# fill ims
		rowidx, colidx = rc
		lagim[rowidx,colidx] = float(shiftval)
		corrim[rowidx,colidx] = float(abs(corr))
	
	return lagim, corrim


def main(stn_id):

	print("=======" * 15)
	print("PROCESSING: {}".format(stn_id))
	print("=======" * 15)

	# Read shape
	stn_gdf = gp.read_file("../shape/{}.shp".format(stn_id))

	# Read runoff
	bf = pd.read_csv("../data/baseflow_sep/baseflow_mm.csv")
	bf['date'] = pd.to_datetime(bf['date'])
	bf.set_index("date", inplace = True)    
	sr = pd.read_csv("../data/baseflow_sep/surface_runoff_mm.csv") 
	sr['date'] = pd.to_datetime(sr['date'])
	sr.set_index("date", inplace = True)   

	# Read rainfall and snowmelt data
	smlt_fn_1d = "../data/Watersheds/1d/{}_1d_smlt.npy".format(stn_id)
	prcp_fn_1d = "../data/Watersheds/1d/{}_1d_prcp.npy".format(stn_id)

	# Set rolling window size
	win_size = 30

	# setup results dir 
	if not os.path.exists("../results"):
		os.mkdir("../results")
	
	# setup outdir 
	outdir = "../results/moving_win_{}".format(str(win_size))
	
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	# Setup outfns 
	prcp_sum_fn = os.path.join(outdir,"{}_prcp_sum.tif".format(stn_id))
	# prcp_mi_fn = os.path.join(outdir,"{}_prcp_mi_{}_{}.tif".format(stn_id)
	prcp_lag_sr_fn = os.path.join(outdir,"{}_prcp_lag_sr.tif".format(stn_id))
	prcp_cor_sr_fn = os.path.join(outdir,"{}_prcp_cor_sr.tif".format(stn_id))
	prcp_lag_bf_fn = os.path.join(outdir,"{}_prcp_lag_bf.tif".format(stn_id))
	prcp_cor_bf_fn = os.path.join(outdir,"{}_prcp_cor_bf.tif".format(stn_id))

	smlt_sum_fn = os.path.join(outdir,"{}_smlt_sum.tif".format(stn_id))
	# smlt_mi_fn = os.path.join(outdir,"{}_smlt_mi_{}_{}.tif".format(stn_id)
	smlt_lag_sr_fn = os.path.join(outdir,"{}_smlt_lag_sr.tif".format(stn_id))
	smlt_cor_sr_fn = os.path.join(outdir,"{}_smlt_cor_sr.tif".format(stn_id))
	smlt_lag_bf_fn = os.path.join(outdir,"{}_smlt_lag_bf.tif".format(stn_id))
	smlt_cor_bf_fn = os.path.join(outdir,"{}_smlt_cor_bf.tif".format(stn_id))


	# Load arrays 
	smlt_1d = np.load(smlt_fn_1d)
	prcp_1d = np.load(prcp_fn_1d)

	prcp_sum = np.nanmean(prcp_1d, axis = 2)
	smlt_sum = np.nanmean(smlt_1d, axis = 2)

	# filter the runoff data to select watershed
	sr_df = sr[stn_id].interpolate(how = 'linear')
	bf_df = bf[stn_id].interpolate(how = 'linear')

	bf_qarr = bf_df.values
	sr_qarr = sr_df.values

	# # if station is PAR, we need to chop off 2017 - 2021
	# if stn_id == "PAR":
	# 	bfdf = bf[stn_id]
	# 	srdf = sr[stn_id]
	# 	mask = (bfdf.index <= "2016-09-30") 
	# 	bf_df = bfdf.loc[mask].interpolate(how = 'linear')
	# 	sr_df = srdf.loc[mask].interpolate(how = 'linear')

	# 	n_days = len(sr_df)
	# 	smlt_1d = smlt_1d[:,:,:n_days]
	# 	prcp_1d = prcp_1d[:,:,:n_days]
	
	# else:
	# 	sr_df = sr[stn_id].interpolate(how = 'linear')
	# 	bf_df = bf[stn_id].interpolate(how = 'linear')

	# 	bf_qarr = bf_df.values
	# 	sr_qarr = sr_df.values


	# CALL MAIN FO PRECIP 

	prcp_bf_lag, prcp_bf_cor = calc_rolling_xcorr(prcp_1d, bf_qarr, win_size = win_size)
	prcp_sr_lag, prcp_sr_cor = calc_rolling_xcorr(prcp_1d, sr_qarr)

	plt.figure(figsize = (12,10))

	plt.subplot(221)
	plt.imshow(prcp_sr_cor); plt.title("prcp - sr corr = {} + / - {}".format(round(np.nanmean(prcp_sr_cor),2), round(np.nanstd(prcp_sr_cor),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(222)
	plt.imshow(prcp_bf_cor); plt.title("prcp - bf corr = {} + / - {}".format(round(np.nanmean(prcp_bf_cor),2), round(np.nanstd(prcp_bf_cor),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(223)
	plt.imshow(prcp_sr_lag); plt.title("prcp - sr lag = {} + / - {}".format(round(np.nanmean(prcp_sr_lag),2), round(np.nanstd(prcp_sr_lag),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(224)
	plt.imshow(prcp_bf_lag); plt.title("prcp - bf lag = {} + / - {}".format(round(np.nanmean(prcp_bf_lag),2), round(np.nanstd(prcp_bf_lag),2)));
	plt.axis("off")
	plt.colorbar()

	plt.tight_layout()
	plt.savefig("../figures/{}_prcp_{}.png".format(stn_id, str(win_size)))

	# Write rasters
	write_raster(prcp_sum, stn_gdf, prcp_sum_fn)
	write_raster(prcp_sr_lag, stn_gdf, prcp_lag_sr_fn)
	write_raster(prcp_sr_cor, stn_gdf, prcp_cor_sr_fn)
	write_raster(prcp_bf_lag, stn_gdf, prcp_lag_bf_fn)
	write_raster(prcp_bf_cor, stn_gdf, prcp_cor_bf_fn)
	# write_raster(smlt_mi, stn_gdf, smlt_mi_fn)

	# CALL MAIN FO SNOWMELT 

	smlt_sr_lag, smlt_sr_cor = calc_rolling_xcorr(smlt_1d, sr_qarr)
	smlt_bf_lag, smlt_bf_cor = calc_rolling_xcorr(smlt_1d, bf_qarr)

	plt.figure(figsize = (12,8))

	plt.subplot(221)
	plt.imshow(smlt_sr_cor); plt.title("smlt - sr corr = {} + / - {}".format(round(np.nanmean(smlt_sr_cor),2), round(np.nanstd(smlt_sr_cor),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(222)
	plt.imshow(smlt_bf_cor); plt.title("smlt - bf corr = {} + / - {}".format(round(np.nanmean(smlt_bf_cor),2), round(np.nanstd(smlt_bf_cor),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(223)
	plt.imshow(smlt_sr_lag); plt.title("smlt - sr lag = {} + / - {}".format(round(np.nanmean(smlt_sr_lag),2), round(np.nanstd(smlt_sr_lag),2)));
	plt.axis("off")
	plt.colorbar()

	plt.subplot(224)
	plt.imshow(smlt_bf_lag); plt.title("smlt - bf lag = {} + / - {}".format(round(np.nanmean(smlt_bf_lag),2), round(np.nanstd(smlt_bf_lag),2)));
	plt.axis("off")
	plt.colorbar()

	plt.tight_layout()
	plt.savefig("../figures/{}_smlt_{}.png".format(stn_id, str(win_size)))


	# Write rasters
	write_raster(smlt_sum, stn_gdf, smlt_sum_fn)
	write_raster(smlt_sr_lag, stn_gdf, smlt_lag_sr_fn)
	write_raster(smlt_sr_cor, stn_gdf, smlt_cor_sr_fn)
	write_raster(smlt_bf_lag, stn_gdf, smlt_lag_bf_fn)
	write_raster(smlt_bf_cor, stn_gdf, smlt_cor_bf_fn)
	# write_raster(smlt_mi, stn_gdf, smlt_mi_fn)


if __name__ == '__main__':

	# Read watersheds
	gdf = gp.read_file("../shape/sierra_catchments.shp")

	stids = list(gdf['stid'].values)

	# main("TRM")

	for stid in stids[:]:
	  main(stid)

	# pool = mp.Pool(3)
	# for i in tqdm(pool.imap_unordered(main, stids), total=len(stids)):
	# 	pass