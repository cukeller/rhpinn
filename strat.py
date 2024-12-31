# create DNN that models log(rho) and log(T) stratification of BiFrost model
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from scales import scales
from readBifrost import readBifrost
from saha_tf import saha
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === adjustable parameters ===
model_name    = 'strat.keras'
epochs        = 20000  # number of epochs
batch_size    = 64     # batch size
learning_rate = 0.02   # learning rate for optimizer

# load BiFrost model
params = ['ux', 'uy', 'uz', 'lgr', 'lgp', 'lgtg']
apar, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, 1, True)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = apar.shape # number of grid points in each axis
ndim = apar.ndim - 1 # number of dimensions; par also has an axis for params
apar = np.moveaxis(apar,1,3)
input_scale, input_offset, output_scale, output_offset = scales()

# density stratification
lgr = np.mean(apar[:,:,:,:,3], axis=(0,1,2))

# temperature stratification
lgt = np.mean(apar[:,:,:,:,5], axis=(0,1,2))

def scaleOffset(input):
    """
    provide offset and scaling for input such that
    output = input * scale + offset
    is in the range -1 to +1
    """
    min = np.amin(input)
    max = np.amax(input)
    scale = 2.0 / (max - min)
    offset = -min * scale - 1.0
    return (input * scale + offset, scale, offset)

z_norm,   z_scale,   z_offset   = scaleOffset(zaxis)
lgr_norm, lgr_scale, lgr_offset  = scaleOffset(lgr)
lgt_norm, lgt_scale, lgt_offset = scaleOffset(lgt)
output_scale  = [1.0/lgr_scale, 1.0/lgt_scale]
output_offset = [-lgr_offset/lgr_scale, -lgt_offset/lgt_scale]


# neural network:
#   output is mean(log10(rho)), mean(log10(T))
#   all outputs are in physical coordinates
model = tf.keras.Sequential([
    tf.keras.layers.Input((1,)),
    tf.keras.layers.Rescaling(scale = z_scale, offset = z_offset),
    tf.keras.layers.Dense(8,  activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(2,  activation='linear'),
    tf.keras.layers.Rescaling(
        scale  = output_scale,
        offset = output_offset
    )
])

X = tf.convert_to_tensor(zaxis, dtype="float32")
Y = tf.convert_to_tensor(np.stack((lgr,lgt), axis = -1), dtype="float32")

lscale = [1e-1, 1e-2]

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.reduce_mean(tf.square(y_true - y_pred), axis = 0) / tf.square(lscale)
    )

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

check_point = tf.keras.callbacks.ModelCheckpoint(
    model_name,
    monitor = 'loss',
    save_best_only = True,
    initial_value_threshold = 1e-3
)

model.fit(
    X,
    Y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1,
    callbacks = [check_point]
)

# print neural network structure
model.summary()

# read best model and predict stratification
best_model = tf.keras.models.load_model(model_name, compile=False)
predicted = best_model.predict(zaxis)
bifrost = [lgr, lgt]
label   = ['lgr', 'lgt']

print('losses per atmospheric parameter')
for i in range(len(label)):
    print(label[i], np.mean((predicted[:,i] - bifrost[i])**2, axis = 0) / lscale[i]**2)

for i in range(len(label)):
    plt.plot(zaxis/1e3, bifrost[i], label = 'Bifrost '+label[i])
    plt.plot(zaxis/1e3, predicted[:,i], label = 'DNN '+label[i])
    plt.legend()
    plt.xlabel('z[km]')
    plt.show()