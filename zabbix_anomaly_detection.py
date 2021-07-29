# Import the libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
from numpy import loadtxt
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Input
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math,json,os,random
from scipy import stats
random.seed(2)
#my scaler
import predict_utils
#read pre-trained scaler
from pickle import load

anomaly_threshold = 5.
tolerance = 20

try:
    # Load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    # Load model
    predict_model = load_model("./Model/conv_ae.h5")
except:
    print("loading error")

# Read Dataframe
data_root_path = "./Dataset"
data5v_path = os.path.join(data_root_path,"predict_data5V.csv")
data6v_path = os.path.join(data_root_path,"predict_data6V.csv")
data5v = predict_utils.read_data(data5v_path,fill_zero=True)
data6v = predict_utils.read_data(data6v_path,fill_zero=True)

# Sliding window generate Input Dataframe
data_list = []
lag = 1
for i in range(0,data5v.shape[0]-100,lag):
    data_list.append(data5v[i:i+100])
for i in range(0,data6v.shape[0]-100,lag):
    data_list.append(data6v[i:i+100])
#--------------------------------------------------------------
# We take shuffle Dataframe for testing
num_list = list(range(len(data_list)))
random.shuffle(num_list) 
for sample_data in  random.sample(data_list,10):

    
    In_data = sample_data
    
    usage = In_data.iloc[:,1:].values
    host = In_data.iloc[:,0].values
    date_time = In_data.index
    
    # Fit scaler with shape (None,8)
    usage = np.concatenate([usage,usage,usage,usage],axis=1)
    usage = scaler.transform(usage)[:,:2]

    # Generate sliding windows
    window_size = 32

    X_list = []
    time_list = []
    host_name = stats.mode(host)[0].item()
    for i in range(0,usage.shape[0]-window_size):
        
        time_list.append(date_time[i:i+window_size])
        X_list.append(usage[i:i+window_size,:])

    # X shape : (window_num,32,2)
    X = np.array(X_list)

    # Predict
    cnt = 0
    anomaly_list = []
    for i in range(X.shape[0]):
        batch = X[i]
        res = predict_model.predict(batch.reshape(1,32,2))

        res = np.concatenate([res,res,res,res],axis=1).reshape(-1,8)
        res = scaler.inverse_transform(res).reshape(-1,32,8)[:,:,:2]
        actual = np.concatenate([batch,batch,batch,batch],axis=1).reshape(-1,8)
        actual = scaler.inverse_transform(actual).reshape(-1,32,8)[:,:,:2]

        score,_,_ = predict_utils.cal_score(actual.reshape(-1,2),res.reshape(-1,2),model_name="Conv")

        if score >= anomaly_threshold:
            cnt += 1
            anomaly_list.append(i)
            
            if len(anomaly_list) >= tolerance:
                
                print("Warning!!")
                print(host_name)
                print("Failure range : {} ~ {}".format(time_list[anomaly_list[0]][0],
                                                       time_list[anomaly_list[-1]][-1]))
                break
        


    # Inverse scaler (num,32,8) -> (num*32,8)
    
    

    

    

    

