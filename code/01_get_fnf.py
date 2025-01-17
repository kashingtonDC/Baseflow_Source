import os
import datetime
import numpy as np
import geopandas as gp
import pandas as pd

def get_fnf(stn_id):
    '''
    Query CA DWR website to get reservoir storage for an area of interest
    '''
    print("**** Fetching FNF for {} ****".format(stn_id))
    
    if stn_id == "SHA":
        stn_id = "SIS"
    elif stn_id == "ORO":
        stn_id = "FTO"
    elif stn_id == "MHB":
        stn_id = "CSN"
    elif stn_id == "EXC":
        stn_id = "MRC"
    elif stn_id == "PNF":
        stn_id = "KGF"
    elif stn_id == "TRM":
        stn_id = "KWT"
    elif stn_id == "ISB":
        stn_id = "KRI"
    elif stn_id == "SJF":
        stn_id = "SBF"
    elif stn_id == "NAT":
        stn_id = "AMF"
        
    url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums=8&dur_code=D&Start=2000-09-01&End=2021-09-01".format(stn_id)
    df = pd.read_csv(url)

    df[stid] = pd.to_numeric(df['VALUE'], errors='coerce').interpolate(how = 'linear') * 0.0283168 * 86400 # cfs --> cms 
    df.index = pd.to_datetime(df['DATE TIME'])
    df.index.names = ['date']
    df.drop(['STATION_ID', "VALUE", "DURATION", "SENSOR_NUMBER", 
             "SENSOR_TYPE", "OBS DATE",'DATE TIME', "DATA_FLAG", "UNITS"], axis = 1, inplace = True)

    df[df[stid] < 0] = np.nan
    
    return df


# Read watersheds 
gdf = gp.read_file("../shape/sierra_catchments.shp")

# Setup out list for dataframes
stn_fnf_dfs= []

for stid in list(gdf['stid'])[:]:

    print(stid + "-------" * 5)
    
    fnf_df = get_fnf(stid)
    stn_fnf_dfs.append(fnf_df)

# concat + write
fin_df = pd.concat(stn_fnf_dfs, axis = 1)
fin_df.to_csv("../data/CDEC/runoff.csv")