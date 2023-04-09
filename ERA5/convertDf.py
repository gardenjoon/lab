import pandas as pd
import numpy as np
from netCDF4 import Dataset
import os
import time
from saveDatas import saveToImg, saveDataFrame, deleteFiles
from transform import timeToStr

def convertToDataFrame(path):
    file_list = os.listdir(path)
    for ncFile in file_list:
        ERA5NC = Dataset(path+ncFile)
        sst_time = time.gmtime(ERA5NC.variables["time"][:][0] + 347122800) # seconds since 1981-01-01 00:00:00
        year = sst_time.tm_year
        month = sst_time.tm_mon
        day = sst_time.tm_mday
        lat = ERA5NC.variables["lat"][:]
        lon = ERA5NC.variables["lon"][:]
        sst = ERA5NC.variables["analysed_sst"][0]

        df = pd.DataFrame()
        for i_lat in range(len(lat)):
            df2 = pd.DataFrame([[s if s != "--" else np.NaN for s in sst[i_lat]]], index=[lat[i_lat]], columns=lon, dtype="float16")
            df = df.append(df2)
        saveDataFrame(df, timeToStr(year, month, day), "./ERA5df/")
    deleteFiles('./ERA5nc/')