from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import numpy as np
from tensorflow.keras import backend as K

# rounded accuracy for the metric
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))
    
def conv_autoencoder(shape1,shape2):

    model = keras.Sequential(
        [
            layers.Input(shape=(shape1, shape2)),
            layers.Conv1D(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=shape2, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    print(model.summary())

    return model

def model_fit(
    X_train_slim,
    X_val_slim,
    codings_size=10,
    conv_layers=1,
    dilation_rate=1,
    activation='relu',
    strides=2,
    seed=31,
    start_filter_no=32,
    kernel_size_1=2,
    epochs=10,
    earlystop_patience=8,
    verbose=2,
    compile_model_only=False,
    batchNormal=False
):

    _, window_size, feat = X_train_slim.shape


    # save the time model training began
    # this way we can identify trained model at the end
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # set random seeds so we can somewhat reproduce results
    tf.random.set_seed(seed)
    np.random.seed(seed)

    end_filter_no = start_filter_no

    inputs = keras.layers.Input(shape=[window_size, feat])
    z = inputs
    
    

    #### ENCODER ####
    for i in range(0, conv_layers):
        z = layers.Conv1D(
                    filters=start_filter_no,
                    kernel_size=kernel_size_1,
                    strides=1,
                    padding="same",                   
                    dilation_rate=1,
                    activation=activation,
                    use_bias=True,
                    
        )(z)
        
        if batchNormal == True:
            z = keras.layers.BatchNormalization()(z)
        z = keras.layers.MaxPool1D(pool_size=2)(z)

    z = keras.layers.Flatten()(z)
    print("Shape of Z:", z.shape)

    bottleneck = keras.layers.Dense(codings_size)(z)

    

    autoencoder_encoder = keras.models.Model(
        inputs=[inputs], outputs=bottleneck
    )

    #### DECODER ####
    decoder_inputs = keras.layers.Input(shape=[codings_size])

    x = keras.layers.Dense(
        start_filter_no * int((window_size / (2 ** conv_layers))), activation="selu"
    )(decoder_inputs)

    x = keras.layers.Reshape(
        target_shape=((int(window_size / (2 ** conv_layers))), end_filter_no)
    )(x)

    for i in range(0, conv_layers):
        x = keras.layers.UpSampling1D(size=2)(x)
        if batchNormal == True:
            x = keras.layers.BatchNormalization()(x)

        x = layers.Conv1D(
                    filters=start_filter_no,
                    kernel_size=kernel_size_1,
                    strides=1,
                    padding="same",                   
                    dilation_rate=1,
                    activation=activation,
                    use_bias=True,
        )(x)

    outputs = keras.layers.Conv1D(
        feat, kernel_size=kernel_size_1, padding="same", activation="sigmoid"
    )(x)
    autoencoder_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

    codings = autoencoder_encoder(inputs)
    reconstructions = autoencoder_decoder(codings)
    convolutional_autoencoder = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

    

    convolutional_autoencoder.compile(
        loss="mse",
        optimizer="adam",  #'rmsprop'
        metrics=[rounded_accuracy],
    )

    # count the number of parameters so we can look at this later
    # when evaluating models
    param_size = "{:0.2e}".format(
        convolutional_autoencoder.count_params() + autoencoder_decoder.count_params()
    )


    # Model Name
    # b : beta value used in model
    # c : number of codings -- latent variables
    # l : numer of convolutional layers in encoder (also decoder)
    # f1 : the starting number of filters in the first convolution
    # k1 : kernel size for the first convolution
    # k2 : kernel size for the second convolution
    # d : whether dropout is used when sampling the latent space (either True or False)
    # p : number of parameters in the model (encoder + decoder params)
    # eps : number of epochs
    # pat : patience stopping number

    model_name = (
        "TBVAE-{}:_str={}_c={}_l={}_f1={}_k1={}_dilr={}"
        "_p={}_eps={}_pat={}".format(
            date_time,
            strides,
            codings_size,
            conv_layers,
            start_filter_no,
            kernel_size_1,
            dilation_rate,
            param_size,
            epochs,
            earlystop_patience,
        )
    )
    
    view_name = (
        "TBVAE-{}:\nstr={}\nc={}\nl={}\nf1={}\nk1={}\ndilr={}"
        "\np={}\neps={}\npat={}".format(
            date_time,
            strides,
            codings_size,
            conv_layers,
            start_filter_no,
            kernel_size_1,
            dilation_rate,
            param_size,
            epochs,
            earlystop_patience,
        )
    )

    print("\n", view_name, "\n")

    if compile_model_only == False:

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0002,
            patience=earlystop_patience,
            restore_best_weights=True,
            verbose=1,
        )

        history = convolutional_autoencoder.fit(
            X_train_slim,
            X_train_slim,
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_data=(X_val_slim, X_val_slim),
            callbacks=[earlystop_callback,],  # tensorboard_callback,
            verbose=verbose,
        )

        return date_time, model_name, history, convolutional_autoencoder, autoencoder_encoder

    else:

        return convolutional_autoencoder, autoencoder_encoder



if __name__ == '__main__':

    conv_autoencoder(32,3)
