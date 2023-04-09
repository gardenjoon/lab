# 경고메세지 제거
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# 그래픽카드 두개 사용
import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])

from autoencoder import *
from saveDatas import *
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def sq_num_ch(sq_num):
    if sq_num == 4:
        return 256
    elif sq_num == 3:
        return 64
    elif sq_num == 2:
        return 16
    elif sq_num == 1:
        return 4

sq_num = 1
dense = 32
day = "20150101"

for sq_num in [1, 2, 3]:
    index_num = sorted(os.listdir(f"../ERA5/squeezed/12875-{sq_num}/")).index(f"{day}.pkl")
    if sq_num == 1:
        index_num = sorted(os.listdir(f"../ERA5/squeezed/12875-{sq_num}/")).index(f"{day}.pkl")
    else:
        index_num = 0
    for dense in [32, 64, 128, 256]:
        try:
            print(f"===========================Start-s{sq_num}-d{dense}===========================")
            channel = 32
            raw_array = readAllDataFramesToArray(f"../ERA5/squeezed/12875-{sq_num}/", start=index_num)
            (total_num, lon, lat) = raw_array.shape
            reshaped = raw_array.reshape(-1, lon, lat, 1)

            if not os.path.isfile(mkPath2(sq_num, dense, channel, "autoencoder", "h5")):
                with mirrored_strategy.scope():
                    (encoder, decoder) = Autoencoder(200, dense, channel, reshaped, sq_num)
            else:
                decoder = tf.keras.models.load_model(mkPath2(sq_num, dense, 32, "autoencoder", "h5"))
        except Exception as e:
            print(e)
            print(f"===========================Failed-s{sq_num}-d{dense}===========================")
            continue


# if not os.path.isfile(f"dec-{day}-s{sq_num}-d{dense}.pkl"):
#     decoded = decoder.predict(reshaped)
#     saveDataFrame(pd.DataFrame(decoded[0].reshape(lon,lat)), "./", f"dec-{day}-s{sq_num}-d{dense}")
# else:
#     decoded = readDataFrame("./", f"dec-{day}-s{sq_num}-d{dense}").values

# raw_array = readDataFrame(f"../ERA5/squeezed/12875-{sq_num}/", day)
# lon = list(raw_array.index)
# lat = list(raw_array.columns)
# raw_array = raw_array.values
# if not os.path.isfile(f"dec-{day}-s{sq_num}-d{dense}.pkl"):
#     values = np.array(decoded[0]).reshape(len(lon),len(lat))
# else:
#     values = decoded
# min_num = np.min(raw_array[raw_array > 0])
# min_n = np.min(values[values > 0])
# max_n = np.max(values)
# values = values * 100
# values[values < min_n+200] = np.NaN
# saveDecoded(list(values), f"dec-{day}-s{sq_num}-d{dense}")
# saveDecoded(raw_array, f"raw-{day}-s{sq_num}-d{dense}")

