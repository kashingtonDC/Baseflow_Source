## This repo contains the code and build instructions for "Identifying Baseflow Source Areas using Ground-Based and Remotely Sensed Hydrologic Data" 

Last tested on Mac OsX sequoia 15.6 in August, 2025

### Step 1 install anaconda


### Build the environmnet 

```
conda create -n plotting python=3.9

conda activate plotting

conda install -c conda-forge matplotlib cartopy rasterio pandas colorcet seaborn shapely geopandas numpy fiona gdal ipykernel jupyter 
```

### Download supporting data files, place in `data` folder 

### Run scripts in order to fetch additional data, process data, and reproduce figures 