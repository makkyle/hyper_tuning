import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import activations
# from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from kerastuner.tuners import RandomSearch
import keras_tuner as kt
import time
from sklearn.model_selection import train_test_split

X1=np.load('data/X1_antigen_antibody_bind__antigen_5688_300_20.npy')
X2=np.load('data/X2_antigen_antibody_bind__light_5688_300_20.npy')
X3=np.load('data/X3_antigen_antibody_bind__heavy_5688_300_20.npy')
Y=np.load('data/Y_antigen_antibody_bind__bind_label_y_5688_1.npy')

print(np.shape(X1))
# time.sleep(100)



x_train = X1
print('x shape',x_train.shape)
y_train = Y
print('y shape',Y.shape)

## split the dataset into training and validation sets
x1_train, x1_val, y_train, y_val = train_test_split(X1,Y,test_size=0.2,random_state=0)
x2_train, x2_val, y_train, y_val = train_test_split(X2,Y,test_size=0.2,random_state=0)
x3_train, x3_val, y_train, y_val = train_test_split(X3,Y,test_size=0.2,random_state=0)

# x_train = np.concatenate((x1_train,x2_train,x3_train))
# x_val = np.concatenate((x1_val,x2_val,x3_val))

# print(np.shape(x_train))
# print(np.shape(x_val))
# print(np.shape(y_train))
# print(np.shape(y_val))
# time.sleep(100)

# n_classes = len(np.unique(y_train))
n_classes = y_train.shape[1]
# print(x_train.shape,x_test.shape)
print("no of classes")
print(n_classes)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res



input_shape = x1_train.shape[1:]
num_transformer_blocks = 4
head_size = 256
mlp_units = [128]


def build_model(hp):

    optimizer_list = ['adam','SGD','adagrad','RMSprop']
    values=[1e-2,1e-3,1e-4,1e-6,1e-7,1e-8]
    lr_min, lr_max = 1e-8, 1e-1

    ## Hyperparameters
    num_heads = hp.Int('num_heads',min_value=2,max_value=12,step=1)

    ff_dim = hp.Int('ff_dim',min_value=4, max_value=64,step=4)

    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=4, max_value=20,step=4)

    dropout = hp.Float('dropout',min_value=0.1, max_value=0.5,step=0.01)

    mlp_dropout = hp.Float('mlp_dropout',min_value=0.1,max_value=0.5,step=0.01)

    # mlp_units = hp.Int('mlp_units',min_value=32,max_value=128,step=16)

    optimizer = hp.Choice('optimizer', optimizer_list)

    lr = hp.Float('learning_rate', min_value=lr_min, max_value=lr_max, sampling='log')

    if optimizer == 'adam':
      
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    elif optimizer == 'SGD':
      
      optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    
    elif optimizer == 'adagrad':
    
      optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    
    elif optimizer == 'RMSprop':
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)



    inputs = keras.Input(shape=input_shape)
    inputs2 = keras.Input(shape=input_shape)
    inputs3 = keras.Input(shape=input_shape)
    x = inputs
    y = inputs2
    z = inputs3

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        y = transformer_encoder(y, head_size, num_heads, ff_dim, dropout)
        z = transformer_encoder(z, head_size, num_heads, ff_dim, dropout)


    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    y = layers.GlobalAveragePooling1D(data_format="channels_last")(y)
    z = layers.GlobalAveragePooling1D(data_format="channels_last")(z)

    x = layers.Concatenate()([x,y,z])
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    transformer_model = keras.Model([inputs,inputs2,inputs3],outputs)
    
    transformer_model.compile(loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],)

    return transformer_model

EPOCHS = 10
# Keras Tuner Stratergy
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=EPOCHS,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# Early Stopping
ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                      patience=4,
                                      verbose=1,
                                      restore_best_weights='True',
                                      min_delta = 0.1
                                     )


# Run Keras Tuner
tuner.search(x=[x1_train,x2_train,x3_train], y=y_train,
             epochs=EPOCHS,
             validation_data=([x1_val,x2_val,x3_val], y_val),
             callbacks=[ES])





