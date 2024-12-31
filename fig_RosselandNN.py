# make figure of Rosseland mean opacity network performance

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import tensorflow as tf

# === adjustable parameters ===
input_file = 'kapp00.ros.txt' # file with opacity table
model_name = 'opann.keras'    # name of PINN model to write

# read number of lines in opacity table file
with open(input_file) as f:
    n = len(f.readlines()) - 2 # two header lines 
f.close()

# empty arrays for input and output to DNN
table = np.zeros((n,2), dtype=np.float32)
kappa = np.zeros((n),   dtype=np.float32)

# read data
with open(input_file) as f:
    # skip two header lines
    f.readline()
    f.readline()
    # read all data into array
    for i in range(0,n):
        line = f.readline()
        table[i,0] = float(line[1:5])  # log10 temperature
        table[i,1] = float(line[5:10]) # log10 pressure
        kappa[i] = float(line[11:17])  # log10 opacity
f.close()

# determine minima, maxima of table
tmin = np.amin(table[:,0])
tmax = np.amax(table[:,0])

pmin = np.amin(table[:,1])
pmax = np.amax(table[:,1])

kmin = np.amin(kappa)
kmax = np.amax(kappa)

# neural network: input is (logT, logP), output is log(kappa), all in CGI units
model = tf.keras.models.load_model(model_name, compile=False)

# table entry minima and maxima
print('minimum and maximum of logT    ', tmin, tmax)
print('minimum and maximum of logP    ', pmin, pmax)
print('minimum and maximum of logKappa', kmin, kmax)

# print neural network structure
model.summary()

# predicted solution over whole space for best model
best_model = tf.keras.models.load_model(model_name, compile=False)
predicted = best_model.predict(table, batch_size = n, verbose = 0)[:,0]
print('RMS residual', math.sqrt(np.mean((kappa - predicted)**2)))

# plot table opacity vs DNN opacity along with residuals
fig = plt.figure()
fig_a = fig.add_subplot(4, 1, (1,3))
fig_a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot([kmin, kmax], [kmin,kmax],c='grey')
plt.scatter(kappa, predicted, s=2.0)
plt.gca().set_xticklabels([])
plt.ylabel('$\\log(\\kappa/\\rho)$ neural network output')
fig_b = fig.add_subplot(4, 1, 4)
fig_b.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig_b.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.plot([kmin,kmax],[0,0],c='grey')
plt.scatter(kappa, kappa - predicted, s=2.0)
plt.xlabel('$\\log(\\kappa/\\rho)$ table input')
plt.ylabel('residuals')
plt.savefig('fig_RosselandNN.pdf', format="pdf", bbox_inches="tight")
