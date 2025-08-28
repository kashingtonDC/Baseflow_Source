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
* [data files](https://cpslo-my.sharepoint.com/:u:/g/personal/aaahamed_calpoly_edu/EWnpe51schJPicfnFmAw15QBVJCBqVt2y-ugsme4OJ8X4w?e=HaEpeo)
* `/data/Watersheds/`

Place in `results` dir: 
* [results files](https://cpslo-my.sharepoint.com/:u:/g/personal/aaahamed_calpoly_edu/EdwYj1OxxjtKgPtXoMQ8OdwBIGF6AReZGW1AcO6omsgI6A?e=QGm39f)
* `/results/bf_te`
* `/results/bf_xc_K180`

#### Next, run Scripts `04-10` process the locally downloaded data and reproduce the figures

### OR Run scripts in order to fetch the raw data data, process data, and reproduce the figures 

Scripts `01-03` fetch and process data. This can take a very long time on a laptop / regular CPU. 
Scripts `04-10` process the locally downloaded data and can be used to reproduce the figures