from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from autoencoder import Autoencoder2
from saveDatas import readAllDataFramesToArray
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
mirrored_strategy = tf.distribute.MirroredStrategy(
    devices=["/GPU:0", "/GPU:1"])

data_path = "../ERA5/squeezed/1800-1/"
# data_path = "../ERA5/ERA5df/"
data_num = 30

raw_array = readAllDataFramesToArray(data_path, data_num)
(total_num, lon, lat) = raw_array.shape
reshaped_data = raw_array.reshape(-1, lon, lat, 1)
with mirrored_strategy.scope():
    Autoencoder2(32, 5, reshaped_data)

# PknuItc2022!