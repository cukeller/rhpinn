# plot z-axis

from zaxis60 import make_zaxis
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


zaxis = make_zaxis()
nz = len(zaxis)
strat = tf.keras.models.load_model('strat62.keras', compile=False)(zaxis)[:,1]

fig = plt.figure(figsize=(10,5))

plt.plot(zaxis/1000.0, strat, linewidth=2.0)
for i in range(nz):
    plt.plot([zaxis[i]/1000.0, zaxis[i]/1000.0], [np.amin(strat), np.amax(strat)],
             linewidth = 0.5, color='grey')

plt.ylabel('$\\log T$')
# plt.ylabel('$\\log\\rho$')
plt.xlabel('z[km]')
# plt.show()
plt.savefig('fig_zaxis.pdf', format="pdf", bbox_inches="tight")
