# creates a neural network that returns the continuum opacity based on an ABSKO kappa table
# author: Christoph U.Keller, ckeller@nso.edu

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === adjustable parameters ===
epochs        = 100000                    # number of epochs
num_batch     = 1                         # number of batches
learning_rate = 0.005                     # learning rate
input_file    = 'opacity_table_500nm.txt' # ABSKO output file with opacities
model_name    = 'opacity_500nm.keras'     # filename of resulting NN model

# read number of lines in opacity file
with open(input_file) as f:
    n = int(f.readline()) # number of data lines is in first line
    print('number of lines ', n)
f.close()

# empty arrays for input and output to DNN
table = np.zeros((n,2), dtype=np.float32)
kappa = np.zeros((n),   dtype=np.float32)

# read opacity data from ABSKO output
with open(input_file) as f:
    f.readline() # skip header line
    # read all data and put into array
    for i in range(0,n):
        line = f.readline()
        table[i,0] = float(line[ 0:10]) # log10 temperature
        table[i,1] = float(line[10:20]) # log10 pressure
        # skipping electron pressure
        kappa[i] =   float(line[30:40]) # log10 opacity
f.close()
print(kappa)

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

# neural network: input is (log10 T, log10 P), output is log10 kappa
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = tf.keras.losses.MeanSquaredError()
)

check_point = tf.keras.callbacks.ModelCheckpoint(
    model_name,
    monitor = 'loss',
    save_best_only = True,
    initial_value_threshold = 2e-4
)

model.fit(
    tf.convert_to_tensor(table, dtype=tf.float32),
    tf.convert_to_tensor(kappa, dtype=tf.float32),
    epochs = epochs,
    batch_size = n // num_batch,
    verbose = 1,
    callbacks = [check_point]
)

# print neural network structure
model.summary()

# predicted solution over whole space for best model
best_model = tf.keras.models.load_model(model_name, compile=False)
predicted = best_model.predict(table, batch_size = n, verbose = 0)[:,0]
print('RMS residual', np.std(kappa - predicted))
# table properties
print('minimum and maximum of logT', tmin, tmax)
print('minimum and maximum of logP', pmin, pmax)
print('minimum and maximum of logKappa', kmin, kmax)

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
