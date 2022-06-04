from sklearn.utils import shuffle
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import traceback,datetime
import numpy as np
import pandas as pd
import AE_TCN,os
import importlib
from IPython.display import clear_output
import matplotlib.pyplot as plt
importlib.reload(AE_TCN)

def read_data(path,o_dtname="Datetime",c_dtname = 'datetime',names=None,skiprows=None):
    df = pd.read_csv(path, sep=',', 
                     parse_dates={c_dtname:[o_dtname]}, 
                     infer_datetime_format=True, 
                     low_memory=False, 
                     na_values=['nan','?'], 
                     index_col=c_dtname,
                     names=names,
                    skiprows=skiprows)

    #df = df.reindex(index=df.index[:,:-1])
    print(df.isnull().sum())
    # filling nan with mean in any columns
    #for j in range(df.shape[1]):        
        #df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
        
    #df.iloc[:,4:] = df.iloc[:,4:]*100
    #another sanity check to make sure that there are not more any nan
    #print(df.isnull().sum())
    #print(df.head())
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

from scipy import stats
def gen_dataset(data,x_window_size,y_window_size):
    X_train = []   #預測點的前 60 天的資料
    y_train = []   #預測點
    for i in range(x_window_size, data.shape[0]-y_window_size,y_window_size):
        X_train.append(data.iloc[i-x_window_size:i, :-1].values)
        
        if np.sum(data.iloc[i-x_window_size:i, -1].values) >= 1:
            y_train.append(1)
        else:
            y_train.append(0)

    X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
    
    #X_train = np.reshape(X_train, (X_train.shape[0], x_window_size,data.shape[1] ))

    """if y_window_size == 1:
        y_train = np.reshape(y_train,(y_train.shape[0], y_train.shape[2]))
    else:
        y_train = np.reshape(y_train,(y_train.shape[0],y_window_size,y_train.shape[2]))"""
        
    print("Gen data info:")
    print("X_data_shape:{}".format(X_train.shape))
    print("y_data_shape:{}".format(y_train.shape))
    print("\n")
          
    return X_train,y_train

def data_scaling(data,X_win_size,y_win_size,mode='robust'):
    
    assert mode == 'robust' or mode == 'minmax','Wrong mode name'
    print("Mode:{}\n".format(mode))
    print(data.describe())
    
    #Record training data index
    data_index = data.shape[0]
   
    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler,RobustScaler
    if mode =='robust':
         #RobustScaler
        scaler = RobustScaler()
        data[list(data.columns)[:-1]] = scaler.fit_transform(data[list(data.columns)[:-1]])     
    elif mode == 'minmax':        
        #MinMaxScaler
        scaler = MinMaxScaler()
        data.iloc[:,:-1] = scaler.fit_transform(data.iloc[:,:-1])     
    
    print(data.describe())
    #generate dataset
    data_X,data_y = gen_dataset(data,X_win_size,y_win_size)
    
    return data_X,data_y,scaler,data,scaler.get_params()   
    
    
# Input the number of iterations you want to search over
random_search_iterations = 250

# random seed value from system input
ransdom_seed_input = 135

# parameters for beta-vae
p_bvae_grid = {
    "latent_sample_rate": [2,4,8,16,32],
    "start_filter_no": [16,32,64,128],
    "nb_stacks": sp_randint(1, 5),
    "kernel_size_1": sp_randint(2, 20),
    "earlystop_patience": sp_randint(20, 50),
    "slide_window_size": [16,32,64],
    "scale_mode": ['robust','minmax']
}

# epochs
epochs = 30

# folder to save models in
model_save_folder = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_mill'

# create the folder
os.mkdir("Model/TCNAE_saved_models/{}".format(model_save_folder))
#(folder_models / 'saved_models' / model_save_folder).mkdir(parents=True, exist_ok=True)

# create dataframe to store all the results
df_all = pd.DataFrame()

# setup parameters to sample
rng = np.random.RandomState(ransdom_seed_input)

# list of parameters in random search
p_bvae = list(ParameterSampler(p_bvae_grid, n_iter=random_search_iterations,random_state=rng))


for i, params in enumerate(p_bvae):
    print('\n### Run no.', i+1)
    
    ### TRY MODELS ###

    # BETA-VAE
    # parameters  
    
    latent_sample_rate =params["latent_sample_rate"] 
    start_filter_no =params["start_filter_no"] 
    kernel_size_1 = params["kernel_size_1"]
    nb_stacks = params["nb_stacks"]
    earlystop_patience = params["earlystop_patience"]
    slide_window_size = params["slide_window_size"]
    scale_mode = params["scale_mode"]

    seed = 16
    verbose = 1
    
    df = read_data("6Vdata_diskio.csv",o_dtname="datetime",c_dtname = 'dt')
    X_WIN = slide_window_size
    Y_WIN = 1
    test_df = df.loc["2021-10-25 12":"2021-10-27 00"]
    df1 = df.loc[:"2021-10-25 12",:]
    df2 = df.loc["2021-10-27 00":,:]
    x_val1 = df1.iloc[7666:]
    x_val2 = df2.iloc[7666:]

    df1_X, df1_y, scaler,df_scale,param = data_scaling(df1,X_WIN,Y_WIN,mode=scale_mode)
    df2_X, df2_y, scaler,df_scale,param = data_scaling(df2,X_WIN,Y_WIN,mode=scale_mode)
    x_val1_X, x_val1_y, scaler,df_scale,param = data_scaling(x_val1,X_WIN,Y_WIN,mode=scale_mode)
    x_val2_X,x_val2_y, scaler,df_scale,param = data_scaling(x_val2,X_WIN,Y_WIN,mode=scale_mode)
    X_test,y_test, scaler,df_scale,param = data_scaling(test_df,X_WIN,Y_WIN,mode=scale_mode)


    X_train = np.concatenate([df1_X,df2_X],axis=0)
    X_val = np.concatenate([x_val1_X,x_val2_X],axis=0)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    os.system("cls")
    


    # try the model and if it doesn't work, go onto the next model
    # not always the best to use 'try' but good enough
    try:

        tcn_ae = AE_TCN.TCNAE(ts_dimension=X_train.shape[2],
                              latent_sample_rate=latent_sample_rate,
                              nb_filters=start_filter_no,
                              kernel_size=kernel_size_1,
                              nb_stacks=nb_stacks,
                              earlystop_patience=earlystop_patience,
                              verbose=1,
                              window_size=slide_window_size)
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = (
        "TCNAE-{}:_c={:.2f}_f={}_k={}_stk={}_er={}_win={}_scale={}".format(
            date_time,
            latent_sample_rate,
            start_filter_no,
            kernel_size_1,
            nb_stacks,
            earlystop_patience,
            slide_window_size,
            scale_mode
            )
        )
        print(model_name)
        history = tcn_ae.fit(X_train,X_train,batch_size=64,epochs=epochs,verbose=1,validation_data=X_val)
        
        # save the model. How to: https://www.tensorflow.org/tutorials/keras/save_and_load
        # save model weights and model json
        model_save_dir_tcnae = os.path.join("Model/TCNAE_saved_models/{}".format(model_save_folder),date_time + "_tcnae")
        # create the save paths
        os.mkdir(model_save_dir_tcnae)

        # save entire bvae model
        
        tcn_ae.save_model(str(model_save_dir_tcnae) + "/tcnae.h5")

        # save encoder bvae model
        """model_as_json = bvae_encoder.to_json()
        with open(r"{}/model.json".format(str(model_save_dir_encoder)), "w",) as json_file:
            json_file.write(model_as_json)
        bvae_encoder.save_weights(str(model_save_dir_encoder) + "/weights.h5")"""

        # get the model run history
        results = pd.DataFrame(history.history)
        epochs_trained = len(results)
        results["epochs_trained"] = epochs_trained
        results = list(
            results[results["val_loss"] == results["val_loss"].min()].to_numpy()
        )  # only keep the top result, that is, the lowest val_loss

        # append best result onto df_model_results dataframe
        if i == 0:
            cols = (
                list(p_bvae[0].keys())
                + list(history.history.keys())
                + ["epochs_trained"]
            )
            results = [[p_bvae[i][k] for k in p_bvae[i]] + list(results[0])]

        else:
            # create dataframe to store best result from model training
            cols = (
                list(p_bvae[0].keys())
                + list(history.history.keys())
                + ["epochs_trained"]
            )
            results = [[p_bvae[i][k] for k in p_bvae[i]] + list(results[0])]

        df = pd.DataFrame(results, columns=cols)

        df["date_time"] = date_time
        df["model_name"] = model_name

        df_all = df_all.append(df, sort=False)

        df_all.to_csv("results_interim_{}.csv".format(model_save_folder))
        
        conv_pre = tcn_ae.predict(X_test)
        conv_pre = scaler.inverse_transform(conv_pre.reshape(-1,4)).reshape(-1,slide_window_size,4)

        """
        for i in range(1,9,1):

            plt.subplot(4,2,i)
            plt.plot(conv_pre[i+1,:,0],color = 'red',marker='.', label = 'Predict MEM Used',linewidth=3)
            plt.plot(X_test[i+1,:,0],color = 'blue',marker='.', label = 'Real MEM Used',linewidth=3)
            plt.grid()
            plt.title("MEM",fontsize=10)

        
        plt.show()
        """
        
        


    except Exception as e:
        print(e)
        print("TRACEBACK")
        traceback.print_exc()
        pass