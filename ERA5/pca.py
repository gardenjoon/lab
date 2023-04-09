from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from time import time, sleep
import saveDatas as sd

def fillNull(data):
    st_fill = time()

    null_index = []
    for i in data.index:
        if data.loc[i].isnull().sum() == len(data.columns):
            null_index.append(i)

    len_data_index = len(data.index)

    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    fit_df = (pd.DataFrame(imputer.fit_transform(data.T))).T

    len_fit_df_index = len(fit_df.index)

    if len_data_index > len_fit_df_index:
        for i in null_index:
            if i > 0:
                fit_df = pd.concat([fit_df.iloc[-1:], fit_df, ], ignore_index=True)
            else:
                fit_df = pd.concat([fit_df, fit_df.iloc[-1:]], ignore_index=True)

    fit_df.index = data.index
    fit_df.columns = data.columns

    return (fit_df, time()-st_fill)

def concatDf(datas, squeeze_num, result_list):
    data_list = []
    time_list = []

    for data_time in datas:
        # filled_df = datas[data_time].fillna(0)
        # filled_df, f_time = fillNull(datas[data_time])

        # flat_arr = datas[data_time].values.ravel()
        flat_arr = datas[data_time].values.ravel()
        if len(flat_arr) == 0: continue
        data_list.append(flat_arr)
        time_list.append(data_time)
    df = pd.DataFrame(data_list, index=time_list, dtype="float64")
    df = df.dropna(axis=1)
    # df.drop(columns = df.columns[df.isna().sum()/len(df) > 0.1], axis=1, inplace=True)
    # df.fillna(df.mean(), inplace=True)
    sd.saveDataFrame(df, "concated-"+str(len(datas))+"-"+str(squeeze_num), "./pca_arrays/")
    print("saved concated"+str(len(datas))+"-"+str(squeeze_num)+".pkl")

    # saveDataFrameToCsv(df,  "", "1-합쳐진데이터")
    # print("saved 1-합쳐진데이터.csv")
    return result_list

def runPCA(dataFrame, times, pca, result_list, name):
    pca_time = time()
    # 공분산행렬
    # scaled = StandardScaler().fit_transform(dataFrame)
    pca_array = pca.fit_transform(dataFrame)
    result_list["PCA"] = round(time()-pca_time, 3)
    sd.saveDataFrame(pd.DataFrame(pca_array, index=times), "pca_array-"+name, "./pca_arrays/")
    print("saved pca_array"+name+".pkl")

    # saveDataFrameToCsv(pd.DataFrame(pca_array, index=times), "", "2-PCA계수")
    # print("saved 2-PCA계수.csv")
    # result = pd.DataFrame({'고윳값':pca.explained_variance_,
    #         '기여율':pca.explained_variance_ratio_},
    #         index=np.array([f"pca{num+1}" for num in range(int(pca.n_components_))]))
    # result['누적기여율'] = result['기여율'].cumsum()
    # saveDataFrameToCsv(result, "", "3-"+str(pca.n_components_)+"PCA결과")
    # print("saved 3-"+str(pca.n_components_)+"PCA결과.csv")

    return (pca, result_list)


def recoverOne(pca_array, path, name, index, pca, min_num):
    df = pca.inverse_transform(pca_array)
    df = pd.DataFrame(df.reshape(min_num, -1))
    sd.saveFigByBaseMap(df, path, name)

def recover(pca_array, path, data_t, name, index, columns, pca):
    recover_time = time()
    df = pca.inverse_transform(pca_array.loc[name])
    recover_time = round(time()-recover_time, 3)

    # df = np.where(df<270, np.NaN, df)
    # df = np.where(df>310, np.NaN, df)

    reshape_time = time()
    df = pd.DataFrame(df.reshape(len(index), len(columns)), index=index)
    reshape_time = round(time()-reshape_time, 5)

    sd.saveFigByBaseMap(df, path, data_t+name)
    return (recover_time, reshape_time, df)
