import os
import datetime
import geopandas as gp
import pandas as pd

def get_fnf(stn_id,start,end):
    '''
    Query CA DWR website to get reservoir storage for an area of interest
    '''
    print("**** Fetching FNF for {} ****".format(stid))

    dt_idx = pd.date_range(start,end, freq='D')
    
    url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations={}&SensorNums=8&dur_code=D&Start=2003-09-01&End=2021-09-01".format(stid)
    df = pd.read_csv(url)
    
    dt_idx = pd.date_range(start,end, freq='D')

    df[stn_id] = pd.to_numeric(df['VALUE'], errors='coerce').interpolate(how = 'linear')
    df.drop(['STATION_ID', "VALUE", "DURATION", "SENSOR_NUMBER", 
             "SENSOR_TYPE", "DATE TIME", "OBS DATE", "DATA_FLAG", "UNITS"], axis = 1, inplace = True)
    
    print(len(dt_idx), len(df))
    df.index = dt_idx
    
    
    return df

# Set start / end
start = datetime.datetime(2003, 9, 1)
end = datetime.datetime(2021, 9, 1)

# Read watersheds 
gdf = gp.read_file("../shape/sierra_catchments.shp")

# Setup out list for dataframes
stn_fnf_dfs= []

# Loop thru watersheds
for stid in list(gdf['stid'])[::-1]:
    try:
        fnf_df = get_fnf(stid, start, end)
        stn_fnf_dfs.append(fnf_df)
    except:
        print(stid + " FAIL")

# concat + write
fin_df = pd.concat(stn_fnf_dfs, axis = 1)
fin_df.to_csv("../data/CDEC/runoff.csv")