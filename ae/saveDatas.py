from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import cartopy.feature as cft
import cartopy.crs as ccrs
import numpy as np
import os
import shutil
import pandas as pd
from time import time, sleep
from sklearn.decomposition import PCA
import joblib
from numba import njit

def saveDataFrame(df, path, name):
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

@njit
def appendByJit(result, data):
    result.append(data)

def readAllDataFramesToArray(path, end = None, start = 0):
    FILES = sorted(os.listdir(path))[start:end]
    result = []
    appender = result.append
    for f in FILES:
        appender(pd.read_pickle(os.path.join(path, f)).fillna(0).values)
    result = np.array(result)
    return result

def readCsv(path, name):
    return pd.read_csv(os.path.join(path, name+".csv"), index_col=0)

def readDataFrame(path, name):
    return pd.read_pickle(os.path.join(path, name+ ".pkl"))

def readDataFrameByTime(time):
    return pd.read_pickle(os.path.join("./ERA5df/", str(time)+ ".pkl"))

def saveDataFrameToCsv(result, path, name):
    return result.to_csv(os.path.join(path, name+".csv"))
    
def saveDecoded(values, name):
    figure = plt.figure(figsize=(12, 10))
    axis_func = plt.axes(projection=ccrs.Robinson())
    # axis_func.coastlines()
    # axis_func.gridlines(linestyle="--", color='black', linewidth=1)
    # axis_func.add_feature(cft.LAND, zorder=100)
    # axis_func.set_global()
    plt.imshow(values, transform=ccrs.PlateCarree(), cmap='jet')
    # color_bar_func = plt.colorbar(
    #     ax=axis_func, orientation="horizontal", aspect=14, shrink=0.8, extend="max")
    # color_bar_func.ax.tick_params(labelsize=10)
    plt.gca().invert_yaxis()

    plt.title(name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./{name}.png")
    plt.close(figure)
    print("\rsaved "+name+'.png')



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
                    transform=ccrs.PlateCarree(), cmap="jet")
    color_bar_func = plt.colorbar(
        ax=axis_func, orientation="horizontal", aspect=14, shrink=0.8, extend="max")
    color_bar_func.ax.tick_params(labelsize=10)

    plt.title(name)
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

    # map = Basemap(projection='cyl', resolution="c", lat_0=0, lon_0=0)
    # map.drawcoastlines()
    # map.drawmapboundary()
    # map.drawmeridians(np.arange(0, 360, 30))
    # map.drawparallels(np.arange(-90, 90, 30))
    # map.fillcontinents(color='white')

    color_bar_func = plt.colorbar(orientation="horizontal", aspect=14, shrink=0.8, extend="max")
    color_bar_func.ax.tick_params(labelsize=10)
    plt.title(name)
    plt.savefig(path+name+'.png')
    plt.close(fig)

def mkPath(dense, channel, name, type_n):
    return "./fig-d"+str(dense)+"-c"+str(channel)+"/"+str(name)+"."+str(type_n)

def mkPath2(sq_num, dense, channel, name, type_n):
    return "./"+str(sq_num)+"/fig-d"+str(dense)+"-c"+str(channel)+"/"+str(name)+"."+str(type_n)

def savePath(dense, channel):
    return "./fig-d"+str(dense)+"-c"+str(channel)+"/"

def saveLossAcc(hist_df, path):
    plt.figure(figsize=(12, 10))
    figs, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist_df['loss'], 'y', label='train loss')
    loss_ax.plot(hist_df['val_loss'], 'r', label='val loss')
    # acc_ax.plot(hist_df['mae']*100,'r',label='mae')
    acc_ax.plot(hist_df['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist_df['val_accuracy'],'g',label='val acc')

    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')

    plt.savefig(f"{path}loss-accuracy.png")
    plt.close()

def savePredImg(decoder, raw_data, target_datas, path, index_list):
    (data_num, lon, lat) = raw_data.shape
    predimg = decoder.predict(target_datas)

    c_index = 0

    for data_time, i in zip(index_list.keys(), index_list.values()):
        fig = plt.figure(figsize=(12, 10))
        rows = 2
        cols = 1

        img1 = raw_data[i]
        min_num = np.min(img1[img1 > 0])
        max_num = np.max(img1)
        img1[img1 <= 0] = np.NaN
        img2 = predimg[c_index].reshape(lon, lat)
        img2[img2 <= min_num] = np.NaN
        img2[img2 > max_num] = np.NaN

        ax1 = fig.add_subplot(rows, cols, 1)
        im = ax1.imshow(img1, cmap='jet')
        ax1.invert_yaxis()
        ax1.set_title('testimg')
        fig.colorbar(im)

        ax2 = fig.add_subplot(rows, cols, 2)
        im2 = ax2.imshow(img2, cmap='jet')
        ax2.invert_yaxis()
        ax2.set_title('predictimg')
        fig.colorbar(im2)

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # cbar = fig.colorbar(im)
        # cbar.ax.tick_params(labelsize=10)

        plt.savefig(f"{path}pred{data_time}.png")
        plt.close()
        c_index += 1

def saveRawImg(t_data, c_data, t_time, c_time, path, name):
    fig = plt.figure(figsize=(12, 10))
    rows = 2
    cols = 1

    ax1 = fig.add_subplot(rows, cols, 1)
    t_im = ax1.imshow(t_data, cmap='jet')
    ax1.invert_yaxis()
    ax1.set_title(t_time)
    fig.colorbar(t_im)

    ax2 = fig.add_subplot(rows, cols, 2)
    c_im = ax2.imshow(c_data, cmap='jet')
    ax2.invert_yaxis()
    ax2.set_title(c_time)
    fig.colorbar(c_im)

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(t_im)
    # cbar.ax.tick_params(labelsize=10)

    plt.savefig(path+name+'.png')
    plt.close(fig)
