import os
import requests
import subprocess

import rasterio as rio

from tqdm import tqdm
from bs4 import BeautifulSoup
from osgeo import gdal

def list_files(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    zipfiles = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    detfiles = [x for x in zipfiles if 'det' in x]
    return detfiles

def download_file(file, data_dir):
    ext = os.path.split(file)[-1]
    outfn = os.path.join(data_dir,ext)
    if not os.path.exists(outfn):
        dl_cmd = 'curl -o {} {}'.format(outfn,file)
        os.system(dl_cmd)
        return outfn
    else:
        return outfn

def untar(file, data_dir):
    tar_cmd = 'tar -xvf {} -C {}'.format(file, data_dir)
    os.system(tar_cmd)
    return file

def clip(intif,outfn, shapefile = "../shape/sierra_catchments.shp"):

    if not os.path.exists(outfn): # Dont write if already there 
        cmd = '''gdalwarp -crop_to_cutline -cutline {} {} {}'''.format(shapefile,intif, outfn)
        print(cmd)
        os.system(cmd)

    return outfn 


def main():

    # main params
    url = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/daily/downloads/'
    ext = 'zip'

    ssebop_dir = "../data/ssebop"
    if not os.path.exists(ssebop_dir):
        os.mkdir(ssebop_dir)

    data_dir = '../data/ssebop/raw'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    cl_dir = "../data/ssebop/clipped"
    if not os.path.exists(cl_dir):
        os.mkdir(cl_dir)

    # setup write dir and check if already existss
    dst_dir = "../data/ssebop/processed"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # List remote files on URL 
    remote_files = list_files(url, ext) 

    # Setup write files 
    clipped_tifs = [os.path.join(cl_dir,os.path.split(x)[1].replace(".zip",'.tif')) for x in remote_files ]

    # main loop 
    for remote_file, clipped_file in tqdm(list(zip(remote_files,clipped_tifs))[:]):

        # specify input and output filenames
        out_file = clipped_file.replace("clipped","processed")

        # check if exists 
        if os.path.exists(out_file):
            print("already processed {} ..... skipping ======= ".format(os.path.split(out_file)[1]))

        # else main processing routine 
        else:
            # download
            localfile = download_file(remote_file, data_dir)

            # print(remote_file)
            # print(localfile)

            # unzip 
            localzip = untar(localfile, data_dir)
            # clean up 
            xml_fn = os.path.splitext(localfile)[0] + ".xml" #.splitext(".zip")
            zip_fn = os.path.splitext(localfile)[0] + ".zip" #.splitext(".zip")
            for oldfile in [zip_fn, xml_fn]:
                if os.path.exists(oldfile):
                    os.remove(oldfile)

            # clip tif and write to clipped dir 
            dst_dir = "../data/ssebop/clipped"
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)

            tif_fn = os.path.splitext(localfile)[0] + ".tif" #.splitext(".zip")
            clipped_tif = clip(tif_fn,clipped_file)

            print(clipped_tif)

            # remove the raw tif 
            if os.path.exists(tif_fn):
                os.remove(tif_fn)

            # Resample the clipped file to the resolution of SNODAS 
            # open reference file and get x and y dimensions 
            ref = rio.open("../results/moving_win_30/merged/prcp_sum.tiff").read(1)
            npix_x = ref.shape[0]
            npix_y = ref.shape[1]

            # Resample            
            resample_cmd = 'gdal_translate -outsize {} {} {} {}'.format(npix_y, npix_x, os.path.abspath(clipped_file), os.path.abspath(out_file))
            os.system(resample_cmd)

            # remove the clipped tif 
            if os.path.exists(clipped_file):
                os.remove(clipped_file)

            print("finished processing {} =======================================".format(out_file))

    return 

if __name__ == '__main__':
    main()