## This repo contains the code and build instructions for "Identifying Baseflow Source Areas using Ground-Based and Remotely Sensed Hydrologic Data" 

Last tested on Mac OsX sequoia 15.6 in August, 2025

Authors: 
* Aakash Ahamed (aaahamed@calpoly.edu), Cal Poly 
* Joseph Janssen, UBC/Stanford
* Rosemary Knight, Stanford

### Step 1 install anaconda

https://www.anaconda.com/download

### Build the environmnet 

clone this repo and `cd` into the directory where you cloned it 

```
conda env create -f environment.yml # This will create a `conda` environment with the required packages
conda activate bfenv # activate the environment
jupyter notebook # launch jupyter to run the notebooks

```

### Download supporting data files, place in relevant dirs

Place in `data` dir: 
* `/data/Watersheds/`:

Place in `results` dir: 
* `/results/bf_te`:
* `/results/bf_xc`:


### Run scripts in order to fetch additional data, process data, and reproduce the figures 

Scripts `01-03` fetch and process data. This can take a very long time on a laptop / regular CPU. 
Scripts `04-10` process the locally downloaded data and can be used to reproduce the figures