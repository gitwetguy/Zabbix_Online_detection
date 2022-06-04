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
from IPython.core.pylabtools import figsize
figsize(15, 7) 
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
plt.plot(data5v.iloc[:,1])
plt.show()