# creates a neural network that returns the Rosseland mean opacity
# author: Christoph U.Keller, ckeller@nso.edu

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === adjustable parameters ===
epochs        = 100000                    # number of epochs
learning_rate = 0.01                      # learning rate
input_file    = 'kapp00.ros.txt'          # Rosseland opacities table
model_name    = 'opacity_rosseland.keras' # filename of resulting NN model

n = 2394 # number of lines in input file
table = np.zeros((n,2), dtype=np.float32)
kappa = np.zeros((n), dtype=np.float32)

# read data
with open(input_file) as f:
    # skip two header lines
    f.readline()
    f.readline()
    # read all data and put normalized version into array
    for i in range(0,n):
        line = f.readline()
        table[i,0] = float(line[1:5])
        table[i,1] = float(line[5:10])
        kappa[i] = float(line[11:17])
f.close()

# determine minima, maxima, scaling and offsets
tmin = np.amin(table[:,0])
tmax = np.amax(table[:,0])
t_scale = 2.0 / (tmax - tmin)
t_offset = -tmin * t_scale - 1.0

pmin = np.amin(table[:,1])
pmax = np.amax(table[:,1])
p_scale = 2.0 / (pmax - pmin)
p_offset = -pmin * p_scale - 1.0

kmin = np.amin(kappa)
kmax = np.amax(kappa)
k_scale = 2.0 / (kmax - kmin)
k_offset = -kmin * k_scale - 1.0

# neural network: input is (logT, logP), output is log(kappa)
model = tf.keras.Sequential([
    tf.keras.layers.Input((2,)),
    tf.keras.layers.Rescaling(
        scale  = [np.float32(t_scale), np.float32(p_scale)], 
        offset = [np.float32(t_offset),np.float32(p_offset)]
    ),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(1, activation='linear'),
    tf.keras.layers.Rescaling(
        scale  = np.float32(1.0/k_scale), 
        offset = np.float32(-k_offset/k_scale)
    )
])

def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(
    optimizer=tf.keras.optimizers.Adam(
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
    table,
    kappa,
    epochs = epochs,
    batch_size = n,
    verbose = 1,
    callbacks = [check_point]
)

# table properties
print('minimum and maximum of logT', tmin, tmax)
print('minimum and maximum of logP', pmin, pmax)
print('minimum and maximum of logKappa', kmin, kmax)

# print neural network structure
model.summary()

# predicted solution over whole space
predicted = model.predict(table, batch_size = n, verbose = 0)
print('RMS residual', math.sqrt(np.mean((kappa - predicted[:,0])**2)))

# plot table opacity vs DNN opacity along with residuals
fig = plt.figure()
fig_a = fig.add_subplot(4, 1, (1,3))
plt.plot([kmin,kmax],[kmin,kmax],c='grey')
plt.scatter(kappa, predicted, s=2.0)
plt.gca().set_xticklabels([])
plt.ylabel('NN log(opacity) output')
fig_b = fig.add_subplot(4, 1, 4)
plt.plot([kmin,kmax],[0,0],c='grey')
plt.scatter(kappa, kappa - predicted, s=2.0)
plt.xlabel('log(opacity) table input')
plt.ylabel('residuals')
plt.show()
