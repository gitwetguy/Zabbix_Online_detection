# Import the libraries
import tensorflow as tf
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, LSTM, TimeDistributed, RepeatVector,GRU, Input, ConvLSTM2D, Bidirectional,BatchNormalization
from tensorflow.keras import Input
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,Nadam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math
import json
from IPython.core.pylabtools import figsize
import solve_cudnn
solve_cudnn.solve_cudnn_error()
figsize(15, 7) 

def read_data(path,fill_zero=False):
    df = pd.read_csv(path, sep=',', parse_dates={'dt':['Datetime']}, infer_datetime_format=True, low_memory=False, na_values=['nan','?'], index_col='dt')

    df = df.reindex(index=df.index[::-1])
    print(df.isnull().sum())
    # filling nan with mean in any columns
    for j in range(df.shape[1]):  
        
        if fill_zero:
        #
            df.iloc[:,j]=df.iloc[:,j].fillna(0)
        else:
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
    #df.iloc[:,4:] = df.iloc[:,4:]*100
    # another sanity check to make sure that there are not more any nan
    print(df.isnull().sum())
    print(df.head())
    return df
#draw_trend
def draw_trend(df,groups,bound=200):
    #figsize(15,10) 
    
    anomly_df = df.copy()
    anomly_df[anomly_df>0] = np.nan
    
    i=1
    Values = df.values
    Anomaly = anomly_df.values
    cols = groups
    
    for group in groups:
        plt.subplot(len(cols), 1, i)
        plt.plot(Values[:bound, group],linewidth=3)
        plt.plot(Anomaly[:bound, group],color='red',linewidth=8)
        plt.title(df.columns[group], y=0.80, loc='right')
        #plt.xticks(range(0,4020,20))
        i += 1
    plt.show()
    print('\n')

def draw_trend_resample(df,groups,resample_freq,bound=200):
    #figsize(15,10) 
    i=1
    Values = df.resample(resample_freq).mean().values
    cols = groups
    for group in groups:
        plt.subplot(len(cols), 1, i)
        plt.plot(Values[:bound, group],linewidth=3)
        #plt.xticks(range(0,bound+10,10))
        plt.title(df.columns[group]+' resample-{}'.format(resample_freq), y=0.80, loc='right')
        i += 1
    plt.tight_layout()
    plt.show()
    
    print('\n')
#draw_corr
def show_corr(df,group):
    plt.matshow(df.iloc[:,group].corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
    plt.title('Spearman_original', size=15)
    plt.xticks(range(len(group)),labels=map(str,group))
    plt.yticks(range(len(group)),labels=map(str,group))
    plt.colorbar()
    plt.show()
    
def show_corr_resample(df,group,resample_freq):
    plt.matshow(df.resample(resample_freq).mean().iloc[:,group].corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
    plt.title('Spearman_resample-{}'.format(resample_freq), size=15)
    plt.xticks(range(len(group)),labels=map(str,group))
    plt.yticks(range(len(group)),labels=map(str,group))
    plt.colorbar()
    plt.show()
    
def gen_dataset(data,y_column_list,x_window_size,y_window_size):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        X_train.append(data[i-x_window_size:i, :])
        y_train.append(data[i:i+y_window_size, y_column_list ])

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    X_train = np.reshape(X_train, (X_train.shape[0], x_window_size,data.shape[1] ))

    if y_window_size == 1:
        y_train = np.reshape(y_train,(y_train.shape[0],y_train.shape[2]))
    else:
        y_train = np.reshape(y_train,(y_train.shape[0],y_window_size,y_train.shape[2]))
        
    print("Gen data info:")
    print("X_data_shape:{}".format(X_train.shape))
    print("y_data_shape:{}".format(y_train.shape))
    print("\n")
            
    return X_train,y_train

def data_scaling(data,X_win_size,y_win_size,mode='robust'):

    assert mode == 'robust' or mode == 'minmax','Wrong mode name'
    print("Mode:{}\n".format(mode))

    #Record training data index
    data_index = data.shape[0]
    feb_index = data.loc['2021-02'].index.shape[0]

    #Y Columns
    y_column=[0,1,2,3,4,5,6,7]



    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler,RobustScaler
    if mode =='robust':
            #RobustScaler
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)     
    elif mode == 'minmax':        
        #MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)     

    #training-test split
    training_set = data_scaled[:feb_index,:]
    testing_set = data_scaled[feb_index:,:]
    print("Train set:{}".format(training_set.shape))
    print("Test set:{}".format(testing_set.shape))

    #generate dataset
    train_X,train_y = gen_dataset(training_set,y_column,X_win_size,y_win_size)
    test_X,test_y = gen_dataset(testing_set,y_column,X_win_size,y_win_size)

    #return  

    return train_X,train_y,test_X,test_y,scaler,data_scaled,scaler.get_params()