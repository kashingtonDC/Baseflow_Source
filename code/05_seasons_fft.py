import os

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gp
import multiprocessing as mp

from osgeo import gdal, osr
from tqdm import tqdm
from sklearn import metrics
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

def cross_correlation_using_fft(x, y):
    f1 = np.fft.fft(x)
    f2 = np.fft.fft(np.flipud(y))
    cc = np.real(np.fft.ifft(f1 * f2))
    return np.fft.fftshift(cc)

def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = abs(zero_index - np.argmax(c))
    return np.argmax(c)


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
    for rc, dfcolidx in zip(rclist,pxdf.columns):

        a=np.ma.masked_invalid(pxdf[dfcolidx])
        b=np.ma.masked_invalid(qarr)
        
        msk = (~a.mask & ~b.mask)
        
        # compute shift and correlation
        try:
            shiftval = compute_shift(a[msk],b[msk])
            # if shiftval <= 0:
            #     shiftval = np.nan
            # if shiftval <0:
            #     shiftval = abs(shiftval)
            # elif shiftval==0:
            #     shiftval = np.nan
            # corrmat = np.ma.corrcoef(a[msk],b[msk])
            # corr = np.nanmean(corrmat[np.where(~np.eye(corrmat.data.shape[0], dtype=bool))].data)
        except:
            shiftval = np.nan

        # try:
        #     corrmat = np.ma.corrcoef(a[msk],b[msk])
        #     corr = np.nanmean(corrmat[np.where(~np.eye(corrmat.data.shape[0], dtype=bool))].data)
        # except:
        #     corr = np.nan

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

    # print(np.nanmean(lagim))
    # print(np.nanmean(corrim))
    # print(np.nanmean(pvalim))

    return lagim, corrim, pvalim

def df_shifted(df, target=None, lag=0):
    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag)
    return  pd.DataFrame(data=new)


def fft_wrapper(imstack,qarr):

    lag_im, corr_im, pval_im = calc_xcorr_fft(imstack,qarr)
    # lag_im, corr_im = calc_xcorr_fft(imstack,qarr)
    # lag_im, corr_im = calc_xcorr(imstack,qarr)

    # Get sum of stack 
    im_sum = np.nansum(imstack, axis = 2)

    # Mask where sum is zero
    lag_im, corr_im = [np.where(im_sum==0, np.nan, x) for x in [lag_im,corr_im]]

    # Mask lag where corr is zero
    # lag_im  = np.where(corr_im<=1e-3, np.nan, lag_im)

    # Filter lag and cor by >0.001 mm tottalthreshold for smlt and >1 for precip 
    # lag_im, corr_im = [np.where(im_sum<np.nanpercentile(im_sum,5), np.nan, x) for x in [lag_im,corr_im]]

    # Filter lag and cor by P value >0.05
    lag_im, corr_im = [np.where(pval_im>0.1, np.nan, x) for x in [lag_im,corr_im]]

    # Mask nans from the pval image 
    lag_im = np.where(np.isnan(pval_im), np.nan, lag_im)

    return lag_im, corr_im, pval_im

def mi_wrapper(imstack,qarr):

    mi_im = calc_mi(imstack,qarr)

    # Get mean of theentire stack 
    im_sum = np.nansum(imstack, axis = 2)

    # mask where sum is zero 
    mi_im = np.where(im_sum==0, np.nan, mi_im)

    # Mask the other nans 
    mi_im = np.where(np.isnan(im_sum), np.nan, mi_im)

    return mi_im

def seasonal_wrapper(stn_id, imstack, shed_ts, hvar):

    # Read watershed gdf
    stn_gdf = gp.read_file("../shape/{}.shp".format(stn_id))


    # Assign seasons to months
    shed_ts['month'] = shed_ts.index.month
    seasons = {10:'F', 11:'F', 12:'F', 1:'W', 2:'W', 3:'W', 4:'Sp', 5:'Sp', 6:'Sp',7:'Su',8:'Su',9:'Su'}
    shed_ts['Season'] = shed_ts['month'].apply(lambda x: seasons[x])

    # Map season to hydro year position
    seas_2_hy = {"F": "1", "W": "2", "Sp":"3", "Su":"4"}

    
    dt_idx = pd.date_range('2003-10-01','2022-09-30', freq='D')
    years = range(2003, 2022)
    for y in tqdm(list(years)[:]):
        ydf = shed_ts[shed_ts.index.year == y]

        for season in list(ydf.Season.unique()):

            # Select the season from year df
            sdf = ydf[ydf.Season==season]

            # Get starting and ending indices of that season and subset data 
            t1 = sdf.index[0]
            t2 = sdf.index[-1]
            window = (dt_idx[dt_idx > t1]& dt_idx[dt_idx <= t2])

            # Copy the df for indices to filter the array
            ts = shed_ts.copy()
            ts['dt'] = ts.index
            ts.reset_index(inplace = True)
            start = ts[ts.dt == window[0]].index
            end = ts[ts.dt == window[-1]].index

            s, e = int(start.values), int(end.values)

            # sum the var during that season
            hvar_sum = np.nansum(imstack[:,:,s:e+1], axis =2)

            # Calc lag, cor, MI 
            print("==========" * 5 )
            print(y, season)
            print("==========" * 5 )

            qarr_in = shed_ts.loc[window][stn_id].interpolate(how = 'linear').values

            # setup outfiles
            seas2hy = {"F":"1","W":"2","Sp":"3","Su":"4"}
            hy_idx = seas2hy[season]
            if season == "F":
                hy = y+1
            else:
                hy = y
                
            # setup seasons dir
            if not os.path.exists("../results/seasons"):
                os.mkdir("../results/seasons")

            # setup outfiles 
            outdir = "../results/seasons/seasonal_timeseries"
            
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            hvar_sum_fn = os.path.join(outdir,"{}_{}_sum_{}_{}.tif".format(stn_id,hvar,hy_idx,str(hy)))
            hvar_pval_fn = os.path.join(outdir,"{}_{}_pval_{}_{}.tif".format(stn_id,hvar,hy_idx,str(hy)))
            # hvar_mi_fn = os.path.join(outdir,"{}_{}_mi_{}_{}.tif".format(stn_id,hvar,hy_idx,str(hy)))
            hvar_lag_fn = os.path.join(outdir,"{}_{}_lag_{}_{}.tif".format(stn_id,hvar,hy_idx,str(hy)))
            hvar_cor_fn = os.path.join(outdir,"{}_{}_cor_{}_{}.tif".format(stn_id,hvar,hy_idx,str(hy)))

            hvar_filelist = [hvar_sum_fn, hvar_lag_fn, hvar_cor_fn]

            # If they don't process and write
            if not all([os.path.isfile(f) for f in hvar_filelist]):
                hvar_lag, hvar_corr, hvar_pval = fft_wrapper(imstack[:,:,s:e+1], qarr_in)
                print('{} lag = {}'.format(hvar, str(np.nanmean(hvar_lag))))

                # hvar_mi = mi_wrapper(imstack[:,:,s:e+1], shed_ts.loc[window][stn_id].interpolate(how = 'linear'))

                write_raster(hvar_lag, stn_gdf, hvar_lag_fn)
                write_raster(hvar_corr, stn_gdf, hvar_cor_fn)
                write_raster(hvar_pval, stn_gdf, hvar_pval_fn)
                # write_raster(hvar_mi, stn_gdf, hvar_mi_fn)
                write_raster(hvar_sum, stn_gdf, hvar_sum_fn)

    return ("WROTE FILES FOR {}".format(stn_id))



def main(stn_id):
    
    print("=======" * 15)
    print("PROCESSING: {}".format(stn_id))
    print("=======" * 15)

    # study period 
    dt_idx = pd.date_range('2003-10-01','2021-09-30', freq='D')

    # Read watershed gdf
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

    smlt = np.load(smlt_fn_1d)
    prcp = np.load(prcp_fn_1d)

    # filter the runoff data to select watershed
    sr_df = pd.DataFrame(sr[stn_id].interpolate(how = 'linear'))
    bf_df = pd.DataFrame(bf[stn_id].interpolate(how = 'linear'))

    # Call wrapper on the scenarios     
    seasonal_wrapper(stn_id, smlt, sr_df, 'smlt_sr')
    seasonal_wrapper(stn_id, smlt, bf_df, 'smlt_bf')

    seasonal_wrapper(stn_id, prcp, sr_df, 'prcp_sr')
    seasonal_wrapper(stn_id, prcp, bf_df, 'prcp_bf')

    print("COMPLETED {}".format(stn_id))


if __name__ == '__main__':

    # Read watersheds
    gdf = gp.read_file("../shape/sierra_catchments.shp")

    stids = list(gdf['stid'].values)

    # main("TRM")

    for stid in stids[:]:
      main(stid)

    # pool = mp.Pool(3)
    # for i in tqdm(pool.imap_unordered(main, stids), total=len(stids)):
    #   pass