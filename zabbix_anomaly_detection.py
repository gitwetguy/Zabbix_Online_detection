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
import math,json,os,random,itertools
import time
from scipy import stats
random.seed(2)
#my scaler
import predict_utils
#read pre-trained scaler
from pickle import load
import tcn
from tcn import TCN 

anomaly_threshold = 5.
tolerance = 20
detection_columns = ["Memory used  (%)",
        "CPU utilization (%) ",
        "Disk (0 C:) - Current Disk Queue Length",
        "Disk (1 D:) - Current Disk Queue Length"]
try:
    # Load the scaler
    scaler = load(open('5v_scaler.pkl', 'rb'))  
except:
    print("loading scaler error")

try:
    # Load model
    predict_model = load_model("./Model/best_model/tcnae.h5",custom_objects={"TCN":TCN})  
except:
    print("loading model error")

def read_data(path,o_dtname="Datetime",c_dtname = 'datetime'):
    df = pd.read_csv(path, sep=',', 
                     parse_dates={c_dtname:[o_dtname]}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['nan','?'], 
                     index_col=c_dtname)
    
    print(df.isnull().sum())
    return df

def pre_process(csv_path):
    

    data = read_data(csv_path,o_dtname="datetime",c_dtname = 'Datetime')
    
    #get Specified name columns
    QoC_df = data[data["name"]=="Disk (0 C:) - Current Disk Queue Length"]
    QoD_df = data[data["name"]=="Disk (1 D:) - Current Disk Queue Length"]
    MEMUSG_df = data[data["name"]=="Memory used  (%)"]
    CPUUSG_df = data[data["name"]=="CPU utilization (%)"]
    
    
    select_col_list = [MEMUSG_df.loc[:,["host","value"]],
                   CPUUSG_df.loc[:,["value"]],
                   QoC_df.loc[:,["value"]],
                   QoD_df.loc[:,["value"]]]

    #Concat dataframe
    new_df = pd.concat(select_col_list,axis=1)
    new_df["host"] = new_df.iloc[:,0]
    new_df["Memory used  (%)"] = new_df.iloc[:,1]
    new_df["CPU utilization (%)"] = new_df.iloc[:,2]
    new_df["Disk (0 C:) - Current Disk Queue Length"] = new_df.iloc[:,3]
    new_df["Disk (1 D:) - Current Disk Queue Length"] = new_df.iloc[:,4]
    new_df = new_df.drop(columns="value")
    

    #Fill NaN
    new_df.fillna(method='ffill', inplace=True)
    new_df.fillna(method='bfill', inplace=True)
    
    #Reverse dataframe if it is reversed 
    if new_df.index[0] > new_df.index[-1]:
        new_df = new_df.iloc[::-1]
    
    #Resample data
    #Usage: Mean within an hour 
    #Quene: Sum within an hour
    new_df = pd.concat([new_df.iloc[:,[0]].resample("h").bfill(),
                        new_df.iloc[:,[1,2]].resample("h").mean(),
                        new_df.iloc[:,[3,4]].resample("h").sum()],
                        axis=1)
    print(new_df)
    return new_df

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def cal_score(y_real,y_hat):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import math

    MAEScore = mean_absolute_error(y_real,y_hat)
    RMSEScore = math.sqrt(mean_squared_error(y_real,y_hat))
    MAPEScore = mean_absolute_percentage_error(y_real,y_hat)

    return round(MAEScore,3),round(RMSEScore,3),round(MAPEScore,3)

df = pre_process("./Dataset/OPTIdata/5v_180days.csv")

df_list = []
for i in range(0,df.shape[0],100):
    
    if df.iloc[i:i+100].shape[0] == 100:
        df_list.append(df.iloc[i:i+100,:])
    else:
        print("tail")
        
        df_list.append(df.iloc[-100:,:])
        
        break

predict_count = 0
window_size = 32
while True:
    
    
    
    if len(df_list) > 0:
        
        if predict_count >= len(df_list):
            print("Waiting...")
            time.sleep(5)
        else:
            df_list[predict_count].iloc[:,[1,2,3,4]] = scaler.transform(df_list[predict_count].iloc[:,[1,2,3,4]])
            
                
            #Sliding Window
            #size=32
            failure_list = list(itertools.repeat(0,df_list[predict_count].shape[0]-window_size))
            time_list = []
            for i in range(0,df_list[predict_count].shape[0]-window_size):
                
                time_range_start = df_list[predict_count].iloc[i:i+window_size].index.values[0]
                time_range_end = df_list[predict_count].iloc[i:i+window_size].index.values[-1]
                
                pre_batch = np.expand_dims(df_list[predict_count].iloc[i:i+window_size,[1,2,3,4]].values,axis=0)
                
                pre = predict_model.predict(pre_batch)
                pre[0] = scaler.inverse_transform(pre[0])
                pre_batch[0] = scaler.inverse_transform(pre_batch[0])
                mae,rmse,_ = cal_score(pre_batch[0,:,0:2],pre[0,:,0:2])
                if (mae+rmse)/2 >=0.1:
                    #print("Anomaly")
                    failure_list[i] = 1
                    time_list.append([str(time_range_start)+"~"+str(time_range_end)])
                    #print("{}~{}".format(time_range_start,time_range_end))
                    #print(*failure_list)
                    
                elif len(time_list) == 0:
                    continue
                else:
                    print("Anomaly Range\n{}~{}".format(str(time_list[0]).split("~")[0],str(time_list[-1]).split("~")[-1]))
                    time_list = []
                    
            
            predict_count += 1
    else:
        print("List empty")        
    