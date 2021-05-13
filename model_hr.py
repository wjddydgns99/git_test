from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,ReLU,BatchNormalization,Activation
from tensorflow.keras.layers import SimpleRNN, Embedding, LSTM,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import os
import glob
import pandas as pd
import dlib
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#모델



maximum = 0
minimum = 200
HR1 = []
for index in range(1,700):
    path_txt = glob.glob('./folder'+str(index)+'/*.csv')
    for i in path_txt:
        df = pd.read_csv(i, encoding='utf-8')
        maximum = max(max(df.loc[:,"PULSE"]),maximum)
        if maximum == 167 : print(index)
        minimum = min(min(df.loc[:,"PULSE"]),minimum)
        temp = df.loc[:, "PULSE"]
        HR1.append(temp)
HR1 = np.array(HR1)
HR_mean = HR1.mean()
print(maximum,minimum,HR_mean)

#def My_Customized_Loss(y_true, y_pred):
 #   return tf.math.reduce_mean(tf.abs(y_pred - HR_mean),axis=-1)

'''
input = Input(batch_input_shape=(1, 25, 30, 3))
CNN = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input)
CNN = BatchNormalization()(CNN)
CNN = Activation('relu')(CNN)
CNN = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(CNN)
CNN = BatchNormalization()(CNN)
CNN = Activation('relu')(CNN)
CNN = MaxPooling2D(strides=2)(CNN)
CNN = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(CNN)
CNN = BatchNormalization()(CNN)
CNN = Activation('relu')(CNN)
CNN = MaxPooling2D(strides=2)(CNN)
CNN = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(CNN)
CNN = BatchNormalization()(CNN)
CNN = Activation('relu')(CNN)
CNN = MaxPooling2D(strides=2)(CNN)
CNN = Reshape((1, 1152))(CNN)
# RNN = Embedding(input_dim=256,output_dim=64)(CNN)
RNN = LSTM(64, return_sequences=True, stateful=True)(CNN)
# RNN=Activation('relu')(RNN)
RNN = Dense(1)(RNN)
model = Model(inputs=input, outputs=RNN)
model = Model(inputs=input, outputs=RNN)
model.compile(optimizer='sgd', loss=tf.keras.losses.MAE, metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.compile(optimizer='sgd', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# model.compile(optimizer='sgd',loss='mse',metrics=['mae'])
print(model.layers[17])


for index in range(1,700):
    #if index == 66:
     #   continue
    HR=[]
    path_txt = glob.glob('./folder'+str(index)+'/*.csv')
    if not path_txt :
        print("No folder"+str(index))
        continue
    print(path_txt)
    path_picture = glob.glob('./folder'+str(index)+'/*.PNG')
    for i in path_txt:
        df = pd.read_csv(i, encoding='utf-8')
        #maximum = max(df.loc[:,"PULSE"])
        #minimum = min(df.loc[:,"PULSE"])
        #print(df.loc[:,"PULSE"])
        #print(maximum, minimum)
        df.loc[:, "PULSE"] = (df.loc[:, "PULSE"] - minimum) / (maximum - minimum) # Min = 40, Max = 150 설정
        df.loc[:, "SPO2"] = df.loc[:, "SPO2"] / 100 # 0 ~ 1.00 으로 Normalization
        temp = df.loc[:,"PULSE"]
        SpO2 = df.loc[:,"SPO2"]
        HR.append(temp)
    HR=np.array(HR)


    #print(HR[0:40])
    #print(HR[0:40])

    x_data=[]
    for i in path_picture:
        #print(i)
        image = cv.imread(i)
        #print(image)
        x_data.append(image)
    if len(x_data) !=40 :
        print("No x_data size is 40")
        continue;
    x_data = np.array(x_data)
    #print(np.array(x_data).shape)
    #x_train = x_data[0:30]
    #y_train = HR[:,0:30]
    #x_val = x_data[30:40]
    #y_val = HR[:,30:40]
    #y_train=y_train.reshape(-1,1)
    #y_val=y_val.reshape(-1,1)
    HR = HR.reshape(-1,1)
    print(x_data.shape)
    print(HR)
    #print(y_train)
    #print(y_val)
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_train)
    #print(y_train)
    #print(np.array(x_train).shape)
    #print(y_train.shape)
    #print(HR[30:40])
    #print(np.array(x_train[0]))
    #print(x_data)
    for epoch in range(50):
        print(index, epoch)
        model.fit(x_data,HR,batch_size=1,shuffle=False)
        #print("---------------------------------------------------------------")
        #print(model.layers[17].states)
        #print("**************************************************************")
        model.layers[17].reset_states()
        #print(model.layers[17].states)

    #print(x_data.shape,HR.shape)
    #print(x_train.shape, y_train.shape)
    #print(x_val.shape, y_val.shape)
#for epoch in range(epochs):
    #for idx in range(11):

model.summary()
        #from tensorflow.keras.utils import plot_model

print(type(model.layers[17]))

#from keras.models import load_model

model.save('epoch_50_Model_Total_700_min_max_normalization_max_is_140.h5')
del model
'''
#from keras.models import load_model
test=490
model = tf.keras.models.load_model('epoch_50_Model_Total_700_min_max_normalization_max_is_140.h5')
print("--------------test----------------")
path_txt = glob.glob('./folder'+str(test)+'/*.csv')
print(path_txt)
HR=[]
for i in path_txt:
    df = pd.read_csv(i, encoding='utf-8')
    #df.loc[:, "PULSE"] = (df.loc[:, "PULSE"] - 40) / (150 - 40)  # Min = 40, Max = 150 설정
    df.loc[:, "SPO2"] = df.loc[:, "SPO2"] / 100  # 0 ~ 1.00 으로 Normalization
    temp = df.loc[:, "PULSE"]
    SpO2 = df.loc[:, "SPO2"]
    HR.append(temp)
HR = np.array(HR)
HR = HR.reshape(-1,1)

path_picture = glob.glob('./folder'+str(test)+'/*.PNG')
print(path_picture)
x_data=[]
for i in path_picture:
     #print(i)
     image = cv.imread(i)
     #print(image)
     x_data.append(image)
if len(x_data) !=40 :
    print("No x_data size is 40")
x_data = np.array(x_data)
import math
y_prediction = model.predict(x_data).flatten()
y_prediction = y_prediction*(maximum-minimum) + minimum
print("예측")
print(y_prediction)
print("측정")
print(HR.flatten())
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#plt.plot(index.value)
plt.xlim(0,40)
plt.ylim(40,150)

plt.plot(y_prediction,lw=1.5,label="Prediction")
plt.plot(HR,lw=1.5,label="Heart Rate")
plt.legend()
plt.xlabel('Second')
plt.ylabel('Heart Rate')
plt.grid(True)
plt.title('HR Estimation')

plt.show()