import os
import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Defining the R script and loading the instance in Python

r = ro.r
r['source']('baseflow_sep.R')

# Loading the function we have defined in R.
baseflow_sep = ro.globalenv['get_baseflow']

# Reading and processing data
df = pd.read_csv("../data/CDEC/runoff.csv")

df.rename(columns = {df.columns[0]: "date"}, inplace = True)
df.set_index('date', inplace = True)

# Setup results_dict
sr_dfs_out = {}
bf_dfs_out = {}

# Loop through cols 
for col in df.columns:
    print(col)

    # Select the column here 
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_pd_df = ro.conversion.py2rpy(df[col].dropna())    
        
    # Call the R function and getting the result
    runoff_df = baseflow_sep(r_pd_df)

    # Converting it back to a pandas dataframe.
    with localconverter(ro.default_converter + pandas2ri.converter):
        outdf = ro.conversion.rpy2py(runoff_df)

    outdf.index = df[col].dropna().index
    # outdf[outdf<0] = 0
    
    bf_dfs_out[col] = outdf['bt']
    sr_dfs_out[col] = outdf['qft']

# Save 
outdir = "../data/baseflow_sep"
if not os.path.exists(outdir):
    os.mkdir(outdir)

bf_outfn = os.path.join(outdir, "baseflow_mm.csv")
if not os.path.exists(bf_outfn):
    baseflow_df.to_csv(bf_outfn)

sr_outfn = os.path.join(outdir, "surface_runoff_mm.csv")
if not os.path.exists(sr_outfn):
    surfrun_df.to_csv(sr_outfn)