import pandas as pd
import numpy as np

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
        scaler = MinMaxScaler(feature_range=(0,1))
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
    
    return train_X,train_y,test_X,test_y,scaler,data_scaled

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def cal_score(y_real,y_hat,model_name):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    import math

    MAEScore = mean_absolute_error(y_real,y_hat)
    RMSEScore = math.sqrt(mean_squared_error(y_real,y_hat))
    MAPEScore = mean_absolute_percentage_error(y_real,y_hat)
        
    #print("Model,MAE,RMSE,MAPE")
    #print("{},{},{},{}".format(model_name,round(MAEScore,3),round(RMSEScore,3),round(MAPEScore,3)))
    #print(round(MAEScore,3))
    
    
    return round(MAEScore,3),round(RMSEScore,3),round(MAPEScore,3)