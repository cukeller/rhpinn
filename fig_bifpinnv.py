# figure comparing directly trained neural network and BiFrost simulations
# vertical slices for (y,x,t,z) axes

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales90 import scales
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'fig_bifpinnv90',
    description = 'Figure comparing Bifrost vertical slices with neural network output'
)
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'neural network model')
parser.add_argument('-o', '--output_figure',
                    type = str,
                    default = 'fig_bifpinnv.pdf',
                    help = 'figure file name')
parser.add_argument('-l', '--label',
                    type = str,
                    default = 'PINN',
                    help = 'label to use for neural network')
parser.add_argument('-zb', '--zoffset_bifrost',
                    type = float,
                    default = 0.0,
                    help = 'z-axis offset to make z=0 equal to tau=1 in Bifrost simulations')
parser.add_argument('-zn', '--zoffset_nn',
                    type = float,
                    default = 0.0,
                    help = 'z-axis offset to make z=0 equal to tau=1 in neural network')

args = parser.parse_args()
model_name =  args.input_model    # name of PINN model to read
output_file = args.output_figure  # name of output figure file
label_nn =    args.label          # label for plot of neural network
zb =          args.zoffset_bifrost
zn =          args.zoffset_nn

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params
input_scale, input_offset, output_scale, output_offset = scales()

y_plot = ny//2
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
        Y[:,i].reshape((ny,nx,nt,nz))[y_plot,:,t_plot,:].min(), 
        par[t_plot,:,y_plot,:,i].min()).min()
    maxs[i] = np.array(
        Y[:,i].reshape((ny,nx,nt,nz))[y_plot,:,t_plot,:].max(), 
        par[t_plot,:,y_plot,:,i].max()).max()

vmax = 4500.0
mins[0] = -vmax
mins[1] = -vmax
mins[2] = -vmax
maxs[0] =  vmax
maxs[1] =  vmax
maxs[2] =  vmax

# figure size in inches
fig = plt.figure(figsize=(8,10))

nrow = 5
ncol = 2
extent_b = (xaxis[0]/1000.0,xaxis[-1]/1000.0,zaxis[0]/1000.0+zb,zaxis[-1]/1000.0+zb)
extent_n = (xaxis[0]/1000.0,xaxis[-1]/1000.0,zaxis[0]/1000.0+zn,zaxis[-1]/1000.0+zn)

par_title = ['$v_x$', '$v_y$', '$v_z$', '$\\log\\rho$', '$\\log T$']
par_cmap  = ['PiYG', 'PiYG', 'seismic', 'inferno', 'afmhot']

for i in range(npar):

    # Bifrost
    fb = fig.add_subplot(nrow, ncol, i * 2 + 1)
    fb.imshow(par[t_plot,::-1,y_plot,:,i], aspect = 'auto',
        interpolation='bicubic', 
        vmin = mins[i], vmax = maxs[i], 
        cmap=par_cmap[i],
        extent=extent_b)
    # fb.axes.invert_yaxis()
    # plt.axis('off')
    if (i==0):
        plt.title('BiFrost')
    plt.ylabel('km')
    # plt.gca().set_yticklabels([])
    if (i < npar-1):
        plt.gca().set_xticklabels([])
    else:
        plt.xlabel('km')

    # PINN
    fn = fig.add_subplot(nrow, ncol, (i + 1) * 2)
    fn.imshow(np.rot90(Y[:,i].reshape((ny,nx,nt,nz))[y_plot,:,t_plot,:]), aspect = 'auto', 
        interpolation='bicubic', 
        vmin = mins[i], vmax = maxs[i], 
        cmap = par_cmap[i],
        extent=extent_n)
    # plt.axis('off')
    if (i==0):
        plt.title(label_nn)
    plt.ylabel(par_title[i])
    plt.gca().set_yticklabels([])
    if (i < npar-1):
        plt.gca().set_xticklabels([])
    else:
        plt.xlabel('km')

# plt.show()
plt.savefig(output_file, format="pdf", bbox_inches="tight")

