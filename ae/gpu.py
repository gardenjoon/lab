# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# from tensorflow.python.client import device_lib
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices("GPU")

# print(gpus)
# print(device_lib.list_local_devices())
# print("GPU Available: ", tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
