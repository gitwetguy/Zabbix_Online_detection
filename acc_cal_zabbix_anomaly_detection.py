# Import the libraries
import tensorflow as tf
from tensorflow import keras
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
import time,traceback
from scipy import stats
random.seed(2)
#my scaler
import predict_utils
#read pre-trained scaler
from pickle import load
import tcn
from tcn import TCN 
from bokeh.plotting import figure, show

#pd.set_option('mode.chained_assignment', None)

anomaly_threshold = 5.
tolerance = 20
detection_columns = ["Memory used  (%)",
        "CPU utilization (%) ",
        "Disk (0 C:) - Current Disk Queue Length",
        "Disk (1 D:) - Current Disk Queue Length"]

try:
    # Load the scaler
    scaler = load(open('6v_minmax_scaler.pkl', 'rb'))
    print("load scaler")  
except:
    print("loading scaler error")

try:
    # Load model
    predict_model = load_model("tcnae.h5",custom_objects={"TCN":TCN})  
    print("load predict model")
except:
    print("loading model error")

#Read CSV 
def read_data(path,o_dtname="dt",c_dtname = 'datetime'):
    df = pd.read_csv(path, sep=',', 
                     parse_dates={c_dtname:[o_dtname]}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['nan','?'], 
                     index_col=c_dtname)
    
    print(df.isnull().sum())
    return df

#Pre process raw data and get key columns
def pre_process(csv_path):
    

    data = read_data(csv_path,o_dtname="dt",c_dtname = 'Datetime')
    
    #get Specified name columns
    QoC_df = data[data["name"]=="Disk (0 C:) - Current Disk Queue Length"]
    QoD_df = data[data["name"]=="Disk (1 D:) - Current Disk Queue Length"]
    MEMUSG_df = data[data["name"]=="Memory used  (%)"]
    CPUUSG_df = data[data["name"]=="CPU utilization (%)"]
    
    #get host name and value
    select_col_list = [MEMUSG_df.loc[:,["host","value"]],
                   CPUUSG_df.loc[:,["value"]],
                   QoC_df.loc[:,["value"]],
                   QoD_df.loc[:,["value"]]]

    #Concat dataframe
    new_df = pd.concat(select_col_list,axis=1)
    new_df.loc[:,"host"] = new_df.iloc[:,0]
    new_df.loc[:,"Memory used  (%)"] = new_df.iloc[:,1]
    new_df.loc[:,"CPU utilization (%)"] = new_df.iloc[:,2]
    new_df.loc[:,"Disk (0 C:) - Current Disk Queue Length"] = new_df.iloc[:,3]
    new_df.loc[:,"Disk (1 D:) - Current Disk Queue Length"] = new_df.iloc[:,4]
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

#caculate mape
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#caculate and return rmse,mae,mape
def cal_score(y_real,y_hat):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import math

    MAEScore = mean_absolute_error(y_real,y_hat)
    RMSEScore = math.sqrt(mean_squared_error(y_real,y_hat))
    #MAPEScore = mean_absolute_percentage_error(y_real,y_hat)

    return MAEScore,RMSEScore

#pre process input data
#df = pre_process("./Dataset/OPTIdata/5v_180days.csv")


df = read_data("./test_df.csv")

#split data into size=100 batch
fixed_win_100 = []
for i in range(0,df.shape[0],100):
    
    if df.iloc[i:i+100].shape[0] == 100:
        fixed_win_100.append(df.iloc[i:i+100,:])
    else:
        print("tail")
        
        fixed_win_100.append(df.iloc[-100:,:])
        
        break

#predict_count: a pointer, if pointer >=  len(list) then Waiting the new data
#window_size: capture time series features using sliding window
#df_list: variables that temporarily store real-time data  
predict_count = 0
window_size = 32
failure_bitmap = []
rmse_list = []
mae_list = []



"""
while True:

    #check
    if X_test.shape[0] > 0 and predict_count < X_test.shape[0]:
        
        #Waiting for data
        if predict_count >= X_test.shape[0]:
            print("Waiting...")
            time.sleep(5)
        
        #anomaly detection
        else:
            
           
            
            
            print(X_test.shape)   
            #Sliding Window
            #size=32
            failure_list = list(itertools.repeat(0,X_test.shape[0]))
            time_list = []
            for i in range(X_test.shape[0]):
                
                
                
                #anomaly detection
                pre = predict_model.predict(X_test[i,:,:].reshape(1,-1,18))
                p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")
                for num in range(X_test.shape[2]):
                    p.line(x=range(X_test.shape[2]),y=X_test[i,:,num])
                show(p)
                print("==============================")
                print("Predict {}th data\n\n".format(i))
                print("==============================")
                time.sleep(0.3)
                os.system("cls")
                pre[0] = scaler.inverse_transform(pre[0])
                X_test[i,:,:] = scaler.inverse_transform(X_test[i,:,:])
                #plt.plot(pre[0,:,0],label="pre")
                #plt.plot(X_test[i,:,0],label="real")
                #plt.yticks(range(0,110,10))
                #plt.legend()
                #plt.show()
                
                
                mae,rmse = cal_score(X_test[i,:,:],pre[0,:,:])
                mae_list.append(mae)
                rmse_list.append(rmse)
                
                
                #error function
                if rmse >= 60000 or mae >= 20000:


                    #print(mae)
                    print("Anomaly in {}th data!!!!".format(i))
                    failure_list[i] = 1
                    #time_list.append([str(time_range_start)+"~"+str(time_range_end)])
                    #print("{}~{}".format(time_range_start,time_range_end))
                    
    
                
                else:
                    #print("Anomaly Range:\n{}~{}".format(str(time_list[0]).split("~")[0],str(time_list[-1]).split("~")[-1]))
                    #print(type(str(time_list[0]).split("~")[0]))
                    #failure_bitmap.append([str(time_list[0]).split("~")[0],str(time_list[-1]).split("~")[-1]])
                    #print(failure_bitmap)
                    time_list = []
                #time.sleep(0.5)
            
            #np.savetxt('failure_bitmap.csv', np.array(failure_bitmap), delimiter=",")
            
            predict_count += 1
            pd.DataFrame(np.array(failure_list)).to_csv("failure_list.csv")
            pd.DataFrame(np.array(mae_list)).to_csv("mae_list.csv")
            pd.DataFrame(np.array(rmse_list)).to_csv("rmse_list.csv")
            break

    else:
        print("List empty")        
"""