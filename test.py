# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:46:45 2019

@author: WJH
"""

import os
import Glo_paras
import keras
from data.data_preprocess import prepro
from model.model import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# LOAD DATA
x_train,y_train,x_val,y_val,x_test,y_test,scaler = prepro(Glo_paras.FILE_PATH)

# INPUT SHAPE
input_shape = (Glo_paras.WINDOW_SIZE,x_train.shape[-1])

# CREATE MODEL
model = BLSTM_model(input_shape)
model.compile(optimizer=keras.optimizers.Adam(1e-4),loss=keras.losses.mean_squared_error)

# LOAD WEIGHTS
ckpt_path = r'D:\学习\科研2019\Temperature-prediction-using-LSTM\weights.60-0.03.h5'
model.load_weights(ckpt_path)

# PREDICT
test_predict = model.predict(x_test,verbose=1,batch_size=Glo_paras.BATCH_SIZE)

# INVERSE SCALER
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# MSE
MSE = mean_squared_error(y_test,test_predict)
print("MSE: ", MSE)

plt.plot(test_predict)
plt.plot(y_test)
plt.legend('predict','true')