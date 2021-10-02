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
from sklearn.model_selection import train_test_split
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
    scaler = load(open('5v_scaler.pkl', 'rb'))
    # Load model
    predict_model = load_model("Model\best_model\tcnae.h5")  
except:
    print("loading error")

def read_data(path,o_dtname="Datetime",c_dtname = 'datetime'):
    df = pd.read_csv(path, sep=',', 
                     parse_dates={c_dtname:[o_dtname]}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['nan','?'], 
                     index_col=c_dtname)
    
    print(df.isnull().sum())
    return df

def pre_process(csv):
    server_name = csv.split("\\")[-1].split("_")[0]
    print(server_name)
    data = read_data(csv,o_dtname="datetime",c_dtname = 'Datetime')
    
    #get Specified name columns
    QoC_df = data[data["name"]=="Disk (0 C:) - Current Disk Queue Length"]
    QoD_df = data[data["name"]=="Disk (1 D:) - Current Disk Queue Length"]
    MEMUSG_df = data[data["name"]=="Memory used  (%)"]
    CPUUSG_df = data[data["name"]=="CPU utilization (%)"]
    
    
    select_col_list = [MEMUSG_df.loc[:,["value"]],
                   CPUUSG_df.loc[:,["value"]],
                   QoC_df.loc[:,["value"]],
                   QoD_df.loc[:,["value"]]]

    new_df = pd.concat(select_col_list,axis=1)
    new_df["Memory used  (%)"] = new_df.iloc[:,0]
    new_df["CPU utilization (%)"] = new_df.iloc[:,1]
    new_df["Disk (0 C:) - Current Disk Queue Length"] = new_df.iloc[:,2]
    new_df["Disk (1 D:) - Current Disk Queue Length"] = new_df.iloc[:,3]
    new_df = new_df.drop(columns="value")
    new_df.to_csv(os.path.join(root_path,"OPTIdata","pre_process_withlabel","{}_server_beforefill.csv".format(server_name)))
    new_df.fillna(method='ffill', inplace=True)
    new_df.fillna(method='bfill', inplace=True)


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

np.save("predict_data.npy",np.array(data_list))

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

        #concat with original data then inverse transform
        res = np.concatenate([res,res,res,res],axis=1).reshape(-1,8)
        res = scaler.inverse_transform(res).reshape(-1,32,8)[:,:,:2]
        actual = np.concatenate([batch,batch,batch,batch],axis=1).reshape(-1,8)
        actual = scaler.inverse_transform(actual).reshape(-1,32,8)[:,:,:2]
        
        
        #caculate anomaly score
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
    
    

    

    

    

