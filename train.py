# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:55:15 2019

@author: WJH
"""
import os
import Glo_paras
import keras
from data.data_preprocess import prepro
from model.model import *

# LOAD DATA
x_train,y_train,x_val,y_val,x_test,y_test,scaler = prepro(Glo_paras.FILE_PATH)

print('Number train samples: ',len(x_train))
print('Number validation samples: ',len(x_val))
print('Number test samples: ', len(x_test))

# INPUT SHAPE
input_shape = (Glo_paras.WINDOW_SIZE,x_train.shape[-1])

# CREATE MODEL
model = BLSTM_model(input_shape)
model.compile(optimizer=keras.optimizers.Adam(1e-4),loss=keras.losses.mean_squared_error)

# CALL BACK
cb_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=Glo_paras.BATCH_SIZE,
                                             write_graph=True, write_grads=True,
                                             write_images=False, embeddings_freq=0,
                                             embeddings_layer_names=None, embeddings_metadata=None)

if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')
cb_ckpt = keras.callbacks.ModelCheckpoint('./checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=1,
                                save_best_only=False, save_weights_only=True, mode='auto', period=10)

cb_es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
# TRAIN MODEL
model.fit(x_train, y_train, batch_size=Glo_paras.BATCH_SIZE, shuffle=True,
          epochs=200, validation_data=(x_val, y_val),
          callbacks=[cb_tensorboard,cb_ckpt,cb_esx ])