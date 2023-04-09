from saveDatas import *
from pca import *
from compare import *
from transform import *
from time import time, sleep
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib
from tabulate import tabulate
import os
# if os.path.exists("1짜리-총걸린시간.csv"):
#     os.remove("1짜리-총걸린시간.csv")
# if os.path.exists("5-최종결과.csv"):
#     os.remove("5-최종결과.csv")
# if os.path.exists("추출결과.csv"):
#     os.remove("추출결과.csv")
# with open("nohup.out", "r+", encoding="utf-8") as f:
#     new_f = f.readlines()
#     f.seek(0)
#     f.truncate()
# 총 불러올 데이터 갯수
data_num = None
if data_num == None:
    data_num = len(os.listdir("./ERA5df"))

# 압축단계 ex)1 -> 1/4  최대 4
squeeze_num = 4

# 샘플 갯수 추출 비율  0 ~ 1 사이
sample_num = 1
fraction = 0.02
if sample_num != None:
    fraction = None
random_state = 1


# 비교 후 추출할 데이터 갯수
extract_num = int(data_num * 10/100)
# extract_num = 

# 최종적으로 저장할 데이터 갯수
final_result_num = 1

isSaveImg = True

# deleteFiles("./raw/")
# deleteFiles("./recovered/")
# deleteFiles("./target/")
# deleteFiles("./pca_arrays/")
# deleteFiles("./squeezed/")

# print("saving squeezed, concated")
# saveSqCo(squeeze_num, data_num)

for x in [0.1, 1, 100]:
    extract_num = int(data_num * x/100)
    for components in [99]:
        for s_num in [4, 3, 2, 1, 0]:
            print("------------------------------")
            print("Start totalDatas: %d components: %d squeeze_num: %d fraction: %f extract_num: %d"%(data_num, components, s_num, round(fraction if fraction!=None else sample_num, 1), extract_num))
            squeeze_path = "./squeezed/"+str(data_num)+"-"+str(s_num)+"/"
            result_list = {
                "pca_components" : 0,
                "squeeze_num" : 0,
                "read" : 0,
                "squeeze" : 0,
                "extract_num": 0,
                # "fill" : 0,
                # "flatten" : 0,
                "concat" : 0,
                "PCA" : 0,
                "recoverImg": 0,
                "extraction": 0,
                "compare" : 0,
                "saveImg" : 0,
                "total" : 0,
                "calc_total" : 0,
            }

            total_result_df = pd.DataFrame()
            if os.path.exists("4-총걸린시간.csv"):
                total_result_df = readCsv("", "4-총걸린시간")
            total_errors = pd.DataFrame()
            if os.path.exists("5-최종결과.csv"):
                total_errors = readCsv("", "5-최종결과")
            total_time = time()

            result_df = pd.DataFrame(result_list, index=[components])
            print("reading...")
            st_read = time()
            raw_datas = readAllDataFrames("./ERA5df/", data_num)
            result_list["read"] = round(time()-st_read, 3)

            print("squeezing...")
            squeeze_time = time()
            datas = raw_datas.copy()
            if s_num != 0:
                if os.path.exists(squeeze_path):
                    datas = readAllDataFrames(squeeze_path)
                else:
                    datas = squeezeAll(raw_datas, s_num)
            result_list["squeeze"] = round(time()-squeeze_time, 3)

            print("concating...")
            concat_time = time()
            if not os.path.isfile("./pca_arrays/concated-"+str(data_num)+"-"+str(s_num)+".pkl"):
                result_list = concatDf(datas, s_num, result_list)
            result_list["concat"] = round(time()-concat_time, 3)

            print("running PCA...")
            pca_path = str(data_num)+"-"+str(components)+"-"+str(s_num)
            if not os.path.isfile("./pca_arrays/pca"+pca_path+".pkl"):
                comp = components / 100
                if components == 100:
                    comp = None
                pca = PCA(n_components=comp)
                concated = readDataFrame("./pca_arrays/", "concated-"+str(data_num)+"-"+str(s_num))
                pca, result_list = runPCA(concated, concated.index, pca, result_list, pca_path)
                joblib.dump(pca, './pca_arrays/pca'+pca_path+'.pkl')
                print("saved pca"+pca_path+'.pkl')
            else:
                pca_time = time()
                pca = joblib.load('./pca_arrays/pca'+pca_path+'.pkl')
                result_list["PCA"] = round(time()-pca_time, 3)

            print("extracting...")
            pca_arrays = readDataFrame("./pca_arrays/", "pca_array-"+pca_path)


            extraction_time = time()
            target_arrays = pca_arrays.sample(sample_num, random_state=random_state)
            lon = datas[list(datas.keys())[0]].index
            lat = datas[list(datas.keys())[0]].columns

            # if isSaveImg == True:
            #     recover_time = time()
            #     for i in target_arrays.index:
            #         if not os.path.exists("./recovered/"+i+"/"):
            #             os.makedirs("./recovered/"+i+"/")
            #         recover(pca_arrays, "./recovered/"+i+"/", "t_"+str(components)+"_"+str(s_num)+"_", i, lon, lat, pca)
            #     result_list["recoverImg"] = round(time()-recover_time, 3)

            for target_time in target_arrays.index:
                for compare_time in pca_arrays.index:
                    if target_time[:4] == compare_time[:4]:
                        pca_arrays.drop(index=compare_time, inplace=True)

            extracted_array = extract(target_arrays, pca_arrays, pca.explained_variance_ratio_, extract_num)
            result_list["extraction"] = round(time()-extraction_time, 3) - result_list["recoverImg"]

            print("last comparing...")
            compare_time = time()
            final_result = {}
            for target_time in extracted_array:
                now_val = raw_datas[target_time]
                sq_now_val = datas[target_time]
                compared_result = []
                for extracted_time in extracted_array[target_time]:
                    compare_val = raw_datas[extracted_time]
                    sq_comp_val = datas[extracted_time]
                    raw_error = int(compare(now_val, compare_val))
                    cos_error, msd_error, pear_error = vari_sim(now_val, compare_val)
                    squeezed_error = None
                    if s_num != 0:
                        squeezed_error = int(compare(sq_now_val, sq_comp_val))
                    else:
                        squeezed_error = raw_error
                    compared_result.append([extracted_time, squeezed_error, raw_error, cos_error, msd_error, pear_error])
                compared_result.sort(key=lambda x: x[2])
                final_result[target_time] = compared_result[:final_result_num]
            result_list["compare"] = round(time()-compare_time, 3)
            if isSaveImg == True:
                print("saving raw pictures...")
                saveImg_time = time()
                for target_time in final_result:
                    saveToImg(raw_datas[target_time], "", target_time)
                    saveToImg(datas[target_time], "", "sq"+target_time)
                    # saveFigByBaseMap(raw_datas[target_time], "./target/", target_time)
                    for result in final_result[target_time]:
                        if not os.path.exists("./raw/"+target_time+"/"):
                            os.makedirs("./raw/"+target_time+"/")
                        result_time = result[0]
                        if not os.path.exists("./recovered/"+target_time+"/"):
                            os.makedirs("./recovered/"+target_time+"/")
                        # recover(pca_arrays, "./recovered/"+target_time+"/", "r_"+pca_path+"_", result_time, lon, lat, pca)
                        saveToImg(raw_datas[result_time], "", result_time)
                        exit()
                        # saveFigByBaseMap(raw_datas[result_time], "./raw/"+target_time+"/", pca_path+"_"+result_time)
                result_list["saveImg"] = round(time()-saveImg_time, 3)
            errors_df = pd.DataFrame(final_result, index=[pca_path for _ in range(len(final_result[target_time]))])
            total_errors = pd.concat([total_errors, errors_df])

            result_list["pca_components"] = pca.n_components_
            result_list["squeeze_num"] = s_num
            result_list["extract_num"] = extract_num

            result_list["total"] = round(time()-total_time, 3)
            result_list["calc_total"] = result_list["extraction"] + result_list["compare"]
            result_df.at[components] = result_list
            result_df = pd.concat([total_result_df, result_df])
            # saveDataFrameToCsv(result_df, "", "4-총걸린시간") 
            saveDataFrameToCsv(total_errors, "", "5-최종결과")
            # print(tabulate(result_df, headers='keys', tablefmt='psql', showindex=True))
            # print(tabulate(total_errors, headers='keys', tablefmt='psql', showindex=True))

# if os.path.exists("1짜리-총걸린시간.csv"):
#     result_df = readCsv("", "1짜리-총걸린시간")
#     print(tabulate(result_df, headers='keys', tablefmt='psql', showindex=True))
# if os.path.exists("5-최종결과.csv"):
#     total_errors = readCsv("", "5-최종결과")
#     print(tabulate(total_errors, headers='keys', tablefmt='psql', showindex=True))
# if os.path.exists("추출결과.csv"):
#     extract_result = readCsv("", "추출결과")
#     print(tabulate(extract_result, headers='keys', tablefmt='psql', showindex=True))