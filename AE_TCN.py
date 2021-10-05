import os,sys
sys.path.append(r"E:\Server_mantain\Spark_test\bioma-tcn-ae-main\src")

import tcnae
import importlib
importlib.reload(tcnae)
from tcnae import TCNAE

def tcn_autoencoder(shape2):
    tcn_ae = TCNAE(ts_dimension=shape2,latent_sample_rate = 4,verbose = 2)
    return tcn_ae
if __name__ == "__main__":
    
    model = tcn_autoencoder(8)
    

