from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cartopy.feature as cft
import cartopy.crs as ccrs
import numpy as np
import os
import shutil
import pandas as pd
from time import time, sleep
from transform import *
from pca import *
from sklearn.decomposition import PCA
import joblib


def saveDataFrame(df, name, path):
    df.to_pickle(os.path.join(path, name + ".pkl"))

def saveAllDataFrames(datas, path):
    for time, value in datas.items():
        value.to_pickle(os.path.join(path, time+".pkl"))

def readAllDataFrames(path, end = None, start = 0):
    FILES = sorted(os.listdir(path))[start:end]
    result = {}
    for f in FILES:
        result[f[:-4]] = pd.read_pickle(os.path.join(path, f))
    return result

def readCsv(path, name):
    return pd.read_csv(os.path.join(path, name+".csv"), index_col=0)

def readDataFrame(path, name):
    return pd.read_pickle(os.path.join(path, name+ ".pkl"))

def readDataFrameByTime(time):
    return pd.read_pickle(os.path.join("./ERA5df/", str(time)+ ".pkl"))

def saveDataFrameToCsv(result, path, name):
    return result.to_csv(os.path.join(path, name+".csv"))

def saveToImg(data, path, name):
    lon = list(data.columns)
    lat = list(data.index)
    values = list(data.values)
    figure = plt.figure(figsize=(12, 10))
    axis_func = plt.axes(projection=ccrs.Robinson())
    axis_func.coastlines()
    axis_func.gridlines(linestyle="--", color='black', linewidth=1)
    axis_func.add_feature(cft.LAND, zorder=100)
    axis_func.set_global()
    plt.contourf(lon, lat, values,
                    transform=ccrs.PlateCarree(), cmap="jet", vmin = 270, vmax = 310)
    # color_bar_func = plt.colorbar(
    #     ax=axis_func, orientation="horizontal", aspect=14, shrink=0.8, extend="max")
    # color_bar_func.ax.tick_params(labelsize=10)

    # plt.title(name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(path + name + '.png')
    plt.close(figure)
    print("\rsaved "+name+'.png')


def saveAllToImg(result_list, datas, path):
    for i in range(len(result_list.index)):
        time = result_list.index[i]
        data = datas[time]
        saveToImg(data, path, str(i)+"_"+time)

def deleteFiles(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def saveFigByBaseMap(data, path, name):
    fig = plt.figure(figsize=(12, 10))
    values = list(data.values)
    lon = list(data.columns)
    lat = list(data.index)
    plt.contourf(lon,lat,values, cmap='jet')

    map = Basemap(projection='cyl', resolution="c", lat_0=0, lon_0=0)
    map.drawcoastlines()
    map.drawmapboundary()
    map.drawmeridians(np.arange(0, 360, 30))
    map.drawparallels(np.arange(-90, 90, 30))
    map.fillcontinents(color='white')

    color_bar_func = plt.colorbar(orientation="horizontal", aspect=14, shrink=0.8, extend="max")
    color_bar_func.ax.tick_params(labelsize=10)
    plt.title(name)
    plt.savefig(path+name+'.png')
    plt.close(fig)
    return name

def saveSqCo(squeeze_num, data_num):
    sq_co_list = {
        "squeeze_num" : 0,
        "read" : 0,
        "squeeze" : 0,
        "concat" : 0,
        "PCA": 0,
    }
    for s_num in range(squeeze_num, 0, -1):
        total_df = pd.DataFrame()
        if os.path.exists("압축시간.csv"):
            total_df = readCsv("", "압축시간")
        sq_co_df = pd.DataFrame(sq_co_list, index=[data_num])
        read_t = time()
        raw_datas = readAllDataFrames("./ERA5df/", data_num)
        sq_co_list["read"] = round(time()-read_t, 3)
        sq_co_list["squeeze_num"] = s_num

        if s_num != 0:
            if not os.path.exists("./squeezed/"+str(data_num)+"-"+str(s_num)+"/"):
                print(str(data_num)+" / "+str(s_num)+" squeezing...")
                sq_t = time()
                datas = squeezeAll(raw_datas, s_num)
                sq_co_list["squeeze"] = round(time()-sq_t, 3)
        if not os.path.isfile("./pca_arrays/concated-"+str(data_num)+"-"+str(s_num)+".pkl"):
            print(str(data_num)+" / "+str(s_num)+" concating...")
            co_t = time()
            if s_num == 0:
                datas = raw_datas.copy()
            else:
                datas = readAllDataFrames("./squeezed/"+str(data_num)+"-"+str(s_num)+"/")
            concatDf(datas, s_num, {})
            sq_co_list["concat"] = round(time()-co_t, 3)
        for components in [80, 95, 99]:
            pca_path = str(data_num)+"-"+str(components)+"-"+str(s_num)
            if not os.path.isfile("./pca_arrays/pca"+pca_path+".pkl"):
                print("calculating "+pca_path+" pca...")
                comp = components / 100
                if components == 100:
                    comp = None
                pca = PCA(n_components=comp)
                concated = readDataFrame("./pca_arrays/", "concated-"+str(data_num)+"-"+str(s_num))
                pca, sq_co_list = runPCA(concated, concated.index, pca, sq_co_list, pca_path)
                joblib.dump(pca, './pca_arrays/pca'+pca_path+'.pkl')

        sq_co_df.at[data_num] = sq_co_list
        sq_co_df = pd.concat([total_df, sq_co_df])
        saveDataFrameToCsv(sq_co_df, "", "압축시간")
