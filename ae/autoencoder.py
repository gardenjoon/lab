from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import BatchNormalization, UpSampling2D, ReLU, Reshape, Flatten, Input, Dense, Conv2D, Conv2DTranspose, AveragePooling2D, PReLU, Concatenate
import numpy as np
import pandas as pd
import tensorflow as tf 
from saveDatas import mkPath, saveLossAcc, savePath, mkPath2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])

def Autoencoder(epochs, dense, channel, train_data, sq_num):
    (train_num, lon, lat, _) = train_data.shape
    encoder_input = Input(shape=(lon, lat, 1))

    pool_num = 1
    if sq_num == 1:
        pool_num = 8
    elif sq_num == 2:
        pool_num = 4
    elif sq_num == 3:
        pool_num = 2

    x = Conv2D(channel, (3, 3), padding='same')(encoder_input)
    x = ReLU()(x)
    x = BatchNormalization()(x)

    x = AveragePooling2D((pool_num, pool_num))(x)
    
    x = Conv2D(channel * 2, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x) 

    x = Flatten()(x)

    encoder_output = Dense(dense)(x)

    encoder = Model(encoder_input, encoder_output)
    encoder.summary()

    decoder_input = Input(shape=(dense, ))

    x = Dense(int(lon/pool_num) * int(lat/pool_num) * dense)(decoder_input)
    x = Reshape((int(lon/pool_num), int(lat/pool_num), dense))(x)

    x = Conv2DTranspose(channel * 2, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((pool_num, pool_num))(x)

    x = Conv2DTranspose(channel, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)

    decoder_output = Conv2DTranspose(1, (3, 3), padding='same')(x)

    decoder = Model(decoder_input, decoder_output)
    decoder.summary()

    encoder_in = Input(shape=(lon, lat, 1))
    x = encoder(encoder_in)
    decoder_out = decoder(x)
    autoencoder = Model(inputs=encoder_in, outputs=decoder_out)
    autoencoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.05), loss='mse', metrics=['accuracy', 'mae'])

    hist = autoencoder.fit(train_data, train_data, epochs=epochs,
                        batch_size=50, shuffle=True, validation_data=(train_data, train_data))

    autoencoder.save(mkPath2(sq_num, dense, channel, "autoencoder", "h5"))
    encoder.save(mkPath2(sq_num, dense, channel, "encoder", "h5"))
    decoder.save(mkPath2(sq_num, dense, channel, "decoder", "h5"))
    return (encoder, decoder)

    # hist_df = pd.DataFrame(hist.history)
    # hist_path = mkPath(dense, channel, "history", "csv")
    # with open(hist_path, mode='w') as f:
    #     hist_df.to_csv(f)
    # hist_df = hist_df.to_dict('list')
    # saveLossAcc(hist_df, savePath(dense, channel))

    return autoencoder

def encoding(encoder, dense, raw_data):
    # step = 100
    # encoded = []
    # if len(raw_data) > step:
    #     start = 0
    #     for end in range(step, len(raw_data), step):
    #         encoded.append(encoder.predict(raw_data[start:end]))
    #         start = end
    #     if end >= len(raw_data)-step:
    #         encoded.append(encoder.predict(raw_data[end:]))
    # else:
    encoded = encoder.predict(raw_data)
    # encoded = np.array(encoded, dtype="float32")
    # encoded = encoded.reshape(-1, dense)
    return encoded

def decoding(decoder, encoded_data):
    # step = 100
    # decoded = []
    # if len(encoded_data) > step:
    #     start = 0
    #     for end in range(start, len(encoded_data), step):
    #         decoded.append(decoder.predict(encoded_data[start:end]))
    #         start = end
    #     if end >= len(encoded_data)-step:
    #         decoded.append(decoder.predict(encoded_data[end:]))
    # else:
    decoded = decoder.predict(encoded_data)
    # encoded = np.array(encoded)
    # encoded = encoded.reshape(-1, 360, 720, 1)
    return decoded
