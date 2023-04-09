import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
from saveDatas import readCsv, saveDataFrameToCsv
def find_min(now, data):
    s = 0
    for lon in list(now.columns):
        for lat in list(now.index):
            s += np.square(now[lon][lat] - data[lon][lat])
    return s

def vari_sim(now_val, compare_val):
    A = now_val
    B = compare_val

    A = np.array(A.values.reshape(-1), dtype=np.float64)
    B = np.array(B.values.reshape(-1), dtype=np.float64)

    compared = np.square(A - B)
    count = pd.DataFrame(compared.reshape(-1)).notnull().sum()
    msd_sim = 1 - (1 / (1 + (np.nansum(compared) / count[0])))

    A = now_val.fillna(0)
    B = compare_val.fillna(0)

    A = np.array(A.values.reshape(-1), dtype=np.float64)
    B = np.array(B.values.reshape(-1), dtype=np.float64)


    pearson_sim = np.dot((A - np.mean(A)), (B - np.mean(B))) / ((np.linalg.norm(A - np.mean(B))) * (np.linalg.norm(B - np.mean(B))))

    cos_sim = dot(A, B)/(norm(A)*norm(B))

    return (round(cos_sim*100, 3), round(msd_sim*100, 3), round(pearson_sim*100, 3))

def compare(now_val, compare_val):
    now_val = np.array(now_val.values, dtype=np.float64)
    compare_val = np.array(compare_val.values, dtype=np.float64)
    compared = np.square(now_val - compare_val)
    return np.nansum(compared)

def sq_extract(now_val, datas, extract_num):
    compared_array = {}
    for compare_time in list(datas.keys()):
        compare_val = datas[compare_time]
        compared_array[compare_time] = int(compare(now_val, compare_val))
    compared_array = dict(sorted(compared_array.items(), key=lambda x: x[1]))
    return list(compared_array.keys())[:extract_num]

def extract(target_df, compare_df, variance, num):
    extracted = {}
    total_extract_result = pd.DataFrame()
    if os.path.exists("추출결과.csv"):
        total_extract_result = readCsv("", "추출결과")
    extract_result = pd.DataFrame()
    for target_t, target_v in target_df.iterrows():
        compared_array = pd.DataFrame(index=compare_df.index)
        target_v = np.array(target_v)
        compare_list = np.array(compare_df.values)
        compared_array[target_t] = (np.square(target_v - compare_list) * variance).sum(axis=1)
        compared_array.sort_values(target_t, inplace=True)
        extracted[target_t] = list(compared_array.index)[:num]
        a = [[index, round(value, 3)] for index, value in zip(compared_array.iloc[:num].index, compared_array.iloc[:num].values.ravel())]
        extract_result = pd.concat([extract_result, pd.DataFrame({target_t: a}, columns=[target_t])], axis=1)
    total_extract_result = pd.concat([total_extract_result, extract_result])
    saveDataFrameToCsv(total_extract_result, "", "추출결과") 
    return extracted

def compareWithPcaRecovered(target_arrays, pca_arrays, pca, num):
    extracted = {}
    for target_t, target_v in target_arrays.iterrows():
        compared_array = pd.DataFrame(index=pca_arrays.index)
        target_v = pca.inverse_transform(target_v)
        compare_list = pca.inverse_transform(pca_arrays)
        compared_array[target_t] = np.square(target_v - compare_list).sum(axis=1)
        compared_array.sort_values(target_t, inplace=True)
        extracted[target_t] = list(compared_array.index)[:num]
    return extracted