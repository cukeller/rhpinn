# figure comparing directly trained neural network and BiFrost simulations
# replaces fig_bifnn.py and fig_bifpinn.py

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales90 import scales
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'fig_bifnn',
    description = 'Figure comparing Bifrost horizontal slices with neural network output'
)
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'neural network model')
parser.add_argument('-o', '--output_figure',
                    type = str,
                    default = 'fig_bifnn.pdf',
                    help = 'figure file name')
parser.add_argument('-l', '--label',
                    type = str,
                    default = 'Neural Network',
                    help = 'label to use for neural network')

args = parser.parse_args()
model_name =  args.input_model     # name of neural network model to read
output_file = args.output_figure  # name of output figure file
label_nn =    args.label

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params
input_scale, input_offset, output_scale, output_offset = scales()

z_plot = nz//2
t_plot = nt//2

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# read NN model
model = tf.keras.models.load_model(model_name, compile=False)

Y = model.predict(X, batch_size = nx*ny*nz)

mins = np.zeros(npar)
maxs = np.zeros(npar)

# determine minimum and maximum values
for i in range(npar):
    mins[i] = np.array(
        Y[:,i].reshape((ny,nx,nt,nz))[:,:,t_plot,z_plot].min(), 
        par[t_plot,z_plot,:,:,i].min()).min()
    maxs[i] = np.array(
        Y[:,i].reshape((ny,nx,nt,nz))[:,:,t_plot,z_plot].max(), 
        par[t_plot,z_plot,:,:,i].max()).max()

# figure size in inches
fig = plt.figure(figsize=(10,4))

nrow = 2
ncol = 5

par_title = ['$v_x$', '$v_y$', '$v_z$', '$\\log\\rho$', '$\\log T$']
par_cmap  = ['PiYG', 'PiYG', 'seismic', 'inferno', 'afmhot']

for i in range(npar):
    fb = fig.add_subplot(nrow, ncol, i + 1)
    fb.imshow(par[t_plot,z_plot,:,:,i], # aspect = aspect, 
        interpolation='bicubic', vmin = mins[i], vmax = maxs[i], cmap=par_cmap[i],
        extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,yaxis[0]/1000.0,yaxis[-1]/1000.0))
    # plt.axis('off')
    if (i==0):
        plt.ylabel('BiFrost')
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.title(par_title[i])

    fn = fig.add_subplot(nrow, ncol, i + 1 + npar)
    fn.imshow(Y[:,i].reshape((ny,nx,nt,nz))[:,:,t_plot,z_plot], # aspect = aspect, 
        interpolation='bicubic', vmin = mins[i], vmax = maxs[i], cmap = par_cmap[i],
        extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,yaxis[0]/1000.0,yaxis[-1]/1000.0))
    # plt.axis('off')
    if (i==0):
        plt.ylabel(args.label)
    plt.gca().set_yticklabels([])
    plt.xlabel('km')

# plt.show()
plt.savefig(output_file, format="pdf", bbox_inches="tight")

