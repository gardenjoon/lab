# 경고메세지 제거
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# 그래픽카드 두개 사용
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])

from autoencoder import *
from saveDatas import *
from compare import *
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from tabulate import tabulate

# data_path = f"./12875-{sq_num}/"
raw_path = "../ERA5/ERA5df/"
data_num = 2000
saveImg = False
sample_num = 100
epochs = 100
encoding_num = 1
final_result_num = 1

def sq_num_ch(sq_num):
    if sq_num == 4:
        return 256
    elif sq_num == 3:
        return 64
    elif sq_num == 2:
        return 16
    elif sq_num == 1:
        return 4

def start(dense, channel, x, sq_num, randint, answer):
    try:
        data_path = f"../ERA5/squeezed/12875-{sq_num}/"
        if sq_num == 0:
            data_path = "./ERA5/ERA5df/"
        extract_num = int(data_num * x / 100)
        save_path = savePath(dense, channel)
        print("================================================================================================")
        print("Start!! Squeeze_num: %d | Dense: %d | Channel: %d | Extract_num: %.1f | Rand_Int: %d"%(sq_num_ch(sq_num), dense, channel, x, randint))
        result_list = {
                "dense" : dense,
                "channel" : channel,
                "extract_num": x,
                "error": 0,
                # "read_t" : 0,
                # "AutoEncoding_t" : 0,
                "encoding_t" : 0,
                "sampling_t": 0,
                "extract_t": 0,
                "compare_t" : 0,
                "total_t" : 0,
                "calc_total" : 0,
            }

        total_result_df = pd.DataFrame()
        if os.path.exists("total_time3.csv"):
            total_result_df = readCsv("", "total_time3")
        total_time = time()
        total_errors = pd.DataFrame()
        if os.path.exists("final_result3.csv"):
            total_errors = readCsv("", "final_result3")
        result_df = pd.DataFrame(result_list, index=[sq_num_ch(sq_num)])

        # print("Reading...")
        # st_read = time()
        # raw_array = readAllDataFramesToArray(data_path, data_num)
        # (total_num, lon, lat) = raw_array.shape
        # reshaped_data = raw_array.reshape(-1, lon, lat, 1)

        # # 오토인코더 실행 or 모델 불러오기
        # print("AutoEncoding...")
        # st_ae = time()
        # if not os.path.isfile(mkPath2(sq_num, dense, channel, "autoencoder", "h5")):
        #     if encoding_num == 1:
        #             Autoencoder(epochs, dense, channel, reshaped_data)
        #     else:
        #             Autoencoder2(dense, channel, epochs, reshaped_data)

        # hist_df = pd.read_csv(mkPath2(sq_num, dense, channel, "history", "csv")).to_dict('list')

        # # 인코더모델 불러오기
        print("Encoding...")
        st_encoding = time()
        encoded_data = readDataFrame("./encoded/", f"s{str(sq_num_ch(sq_num))}-d{str(dense)}-c{str(channel)}")
        encoded_data = encoded_data.values[:data_num].tolist()
        result_list["encoding_t"] = round((time()-st_encoding), 3)

        # 랜덤으로 타깃데이터 선택
        print("Sampling...")
        st_sampling = time()
        time_list = []
        extract_data = {}
        for i, filename in enumerate(natsorted(os.listdir(data_path))[:data_num]):
            data_t = os.path.splitext(filename)[0]
            time_list.append(data_t)
            extract_data[data_t] = encoded_data[i]
        extract_df = pd.DataFrame(extract_data).swapaxes(axis1=0, axis2=1)
        target_df = extract_df.sample(sample_num, random_state=randint)

        index_list = {}
        for target_time in target_df.index:
            index_list[target_time] = list(extract_df.index).index(target_time)
        for target_time in target_df.index:
            for compare_time in extract_df.index:
                if target_time[:6] == compare_time[:6]:
                    extract_df.drop(index=compare_time, inplace=True)
        result_list["sampling_t"] = round((time()-st_sampling), 3)


        # 데이터 추출
        print("Extracting...")
        st_extract = time()
        extracted = extract(target_df, extract_df, extract_num)
        result_list["extract_t"] = round((time()-st_extract), 3)


        # 추출된 데이터와 원본비교
        print("Comparing...")
        st_compare = time()
        raw_datas = readAllDataFrames(raw_path, data_num)
        final_result = {}
        for target_time in extracted:
            now_val = raw_datas[target_time]
            compared_result = []
            for extracted_time in extracted[target_time]:
                compare_val = raw_datas[extracted_time]
                raw_error = int(compare(now_val, compare_val))
                compared_result.append([extracted_time, raw_error])
            compared_result.sort(key=lambda x: x[1])
            final_result[target_time] = (compared_result[:final_result_num])[0][1]
            if x != 10:
                final_result[target_time] = np.abs(answer[target_time] - final_result[target_time])
        if x == 10:
            print("Answer Calc Succeed")
            return final_result
        result_list["compare_t"] = round(time()-st_compare, 3)

        errors_df = pd.DataFrame(final_result, index=[x])
        total_errors = pd.concat([total_errors, errors_df])

        result_list["total_t"] = round(time()-total_time, 3)
        result_list["calc_total"] =  result_list["extract_t"] + result_list["compare_t"]
        
        result_df.loc[sq_num_ch(sq_num)] = result_list
        result_df = pd.concat([total_result_df, result_df])

        print("Saving Results...")
        saveDataFrameToCsv(result_df, "", "total_time3") 
        saveDataFrameToCsv(total_errors, "", "final_result3")

        # 이미지 저장
        if saveImg:
            print("Saving Images...")
            saveLossAcc(hist_df, save_path)
            if encoding_num == 1:
                decoder = tf.keras.models.load_model(mkPath2(sq_num, dense, channel, "decoder", "h5"))
                savePredImg(decoder, raw_array, target_df.values, save_path, index_list)
            else:
                autoencoder = tf.keras.models.load_model(mkPath2(sq_num, dense, channel, "autoencoder", "h5"))
                savePredImg(autoencoder, raw_array, target_df.values, save_path, index_list)

        return (result_df)
    except Exception as e:
        print(e)
        exit()


for count in range(1, 101):
    print(f"=================================  {count}  =====================================")
    print(f"=================================  {count}  =====================================")
    print(f"=================================  {count}  =====================================")
    print(f"=================================  {count}  =====================================")
    print(f"=================================  {count}  =====================================")
    random = np.random.randint(100000, size=1)[0]
    (answer) = start(256, 64,10, 3, random, 1)
    for sq_num in [4, 3, 2, 1]:
        for dense in [64, 128, 256]:
            for channel in [32]:
                print(f"{count} 번째")                
                for x in [0.1, 1]:
                    try:
                        (result_df) = start(dense, channel, x, sq_num, random, answer)
                    except:
                        continue

print(tabulate(result_df, headers='keys', tablefmt='psql', showindex=True))

# nohup python3 -u run2.py > ./nohup2.out 2>&1 &
