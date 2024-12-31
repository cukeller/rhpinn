# Figure showing quantitative PINN model 90 comparison with BiFrost model
# uses correlation
# has axis order (y,x,t,z)

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales90 import scales
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'fig_corr',
    description = 'Quantitative comparison of radiative hydrodynamics neural network with Bifrost simulations'
)
parser.add_argument('-i', '--input_model',
                    required = True,
                    type = str,
                    help = 'neural network model')
parser.add_argument('-o', '--output_figure',
                    type = str,
                    default = 'fig_corr.pdf',
                    help = 'figure file name')
parser.add_argument('-zb', '--zoffset_bifrost',
                    type = float,
                    default = 0.0,
                    help = 'z-axis offset in km to make z=0 equal to tau=1 in Bifrost simulations')
args = parser.parse_args()

# === adjustable parameters ===
model_name =  args.input_model    # name of neural network model to read
output_file = args.output_figure  # name of output figure file
zb =          args.zoffset_bifrost

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1, True)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params
input_scale, input_offset, output_scale, output_offset = scales()
print('nx,ny,nz,nt', nx,ny,nz,nt)

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis-zb, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# read PINN model
model = tf.keras.models.load_model(model_name, compile=False)

# move axes in BiFrost to (y,x,t,z)
bif = np.moveaxis(par,[0,1],[2,3]).reshape(-1, nz, npar)

# calculate PINN at BiFrost grid points
mod = model.predict(X, batch_size = nx * nz * 4).reshape(-1, nz, npar + 4)[...,:npar]

# calculate correlation between PINN and BiFrost as a function of height
bif_0mean = bif - np.mean(bif, axis=0, keepdims=True)
mod_0mean = mod - np.mean(mod, axis=0, keepdims=True)
corr = np.mean(mod_0mean * bif_0mean, axis=0) / np.std(mod_0mean, axis=0) / np.std(bif_0mean, axis=0)

line_style = [':', ':', '-', '--', '-.']
line_label = ['$v_x$', '$v_y$','$v_z$','$\\log\\rho$','$\\log T$']

zstart = 24
zend   = 45
# zstart = 0
# zend   = nz
for i in range(npar):
    plt.plot(zaxis[zstart:zend] / 1000.0 + 108.0, corr[zstart:zend,i], label = line_label[i], ls=line_style[i])
plt.legend(prop={'family':'Times New Roman'})
plt.xlabel('height [km]')
# plt.ylabel('R(PINN,Bifrost)')
plt.ylabel('R(DNN,Bifrost)')

# plt.show()
plt.savefig(output_file, format="pdf", bbox_inches="tight")
