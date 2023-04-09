import saveDatas as sd
import pandas as pd
import numpy as np
import os

def squeeze(data, num):
    df = data.copy()
    for _ in range(num):
        df = pd.concat([df.loc[:0], df.loc[0:]])
        lon = np.repeat([df.columns[i]
                        for i in range(0, len(df.columns), 2)], 2)
        lat = np.repeat([df.index[i] for i in range(0, len(df.index), 2)], 2)

        df = df.swapaxes(axis1=0, axis2=1)
        df.set_axis(labels=lon, axis=0, inplace=True)
        df = df.groupby(level=0).mean()

        df = df.swapaxes(axis1=0, axis2=1)
        df.set_axis(labels=lat, axis=0, inplace=True)
        df = df.groupby(level=0).mean()
    return df


def squeezeAll(datas, squeezeNum):
    path = "./squeezed/"+str(len(datas))+"-"+str(squeezeNum)+"/"
    if not os.path.exists(path):
        os.makedirs(path)

    df = datas.copy()
    for key, value in df.items():
        df[key] = squeeze(value, squeezeNum)
        sd.saveDataFrame(df[key], key, path)
    return df

def readSqueezed(path):
    squeezed = sd.readDataFrame(path, name)
    df = {}
    for data_time in squeezed.columns:
        concat_df = {}
        for i in range(len(squeezed[data_time].index)):
            concat_df[squeezed[data_time].index[i]] = squeezed[data_time].values[i]
        concat_df = pd.DataFrame(concat_df)
        df[data_time] = concat_df
    return df

def timeToStr(year, month, day, dash=False):
    month = month if month >= 10 else "0" + str(month)
    day = day if day >= 10 else "0" + str(day)
    if dash == True:
        return str(year)+"-"+str(month)+"-"+str(day)
    else:
        return str(year)+str(month)+str(day)

def showResult(pca_array, pca):
    result = pd.DataFrame({'고윳값':pca.explained_variance_,
                 '기여율':pca.explained_variance_ratio_},
                index=np.array([f"pca{num+1}" for num in range(components)]))
    result['누적기여율'] = result['기여율'].cumsum()
    print(result)

