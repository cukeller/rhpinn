# Calculate continuum images from DNN
# - DNN with coordinates in (y,x,t,z) order
# - uses irregularly spaced z-axis
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
from saha_tf import saha
from zaxis import make_zaxis
import plot
import math
import os
import argparse
import numpy as np
from astropy.io import fits
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'contimag',
    description = 'Calculate images from deep neural network hydrodynamics models'
)
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'neural-network model')
parser.add_argument('-o', '--opacity',
                    type = str,
                    default = 'Opacity/opacity_500nm.keras',
                    help = 'opacity neural-network')
parser.add_argument('-d', '--directory_images',
                    type = str,
                    default = 'imag_bifrost',
                    help = 'directory where images are written to')
args = parser.parse_args()

# === adjustable parameters ===
model_name =   args.input_model      # name of DNN model to read
opacity_name = args.opacity          # file name of opaciity neural network     
image_dir =    args.directory_images # name of directory where images should be stored

# === constants ===
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, 1, False)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions, par is fastest axis

# z-axis with irregular spacing
zaxis = make_zaxis()
nz = len(zaxis)

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((ym, xm, tm, zm), axis = -1).reshape(nx*ny*nz*nt, ndim),
    dtype="float32"
)

# load continuum opacity neural network
opacity = tf.keras.models.load_model(opacity_name, compile=False)
# load DNN describing atmosphere
model = tf.keras.models.load_model(model_name, compile=False)

# Determine model parameters from DNN
Y = model.predict(X, batch_size = nx*ny*nz//4, verbose = 0)
lr = Y[:,3]
lt = Y[:,4]
tgas = np.power(10.0, lt)
rho =  np.power(10.0, lr)

# hydrogen ionization fraction
ionfrac = saha(tgas, rho)

# gas pressure EOS
lnp = (lr + lt) * math.log(10) + 8.7711096 + np.log(1.0 + ionfrac * 0.934)
pgas = np.exp(lnp)

# opacity; opacity() works in CGS units
kappa = np.power(10.0, opacity(np.stack((lt, lnp / math.log(10) + 1), axis=1))[:,0] - 1)

# from scipy
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def cumtrapz(y, x, axis=-1, initial=None):
    d = np.diff(x)
    # reshape to correct shape
    shape = [1] * y.ndim
    shape[axis] = -1
    d = d.reshape(shape)

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = np.add.accumulate(d * (y[slice1] + y[slice2]) / 2.0, axis)

    if initial is not None:
        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.ones(shape, dtype=res.dtype) * initial, res], axis=axis)

    return res

# optical depth: integrating opacity from top to bottom in z-axis using trapezoid rule
tau = cumtrapz(
    np.reshape(kappa*rho, (-1,nz))[...,::-1], 
    -zaxis[::-1], 
    initial = 1e-7
)[...,::-1]

c =   1.191066E-5 # 2*h*c^2
hck = 1.438832334 # h*c/k
wl =  500e-7 # wavelength in cm
# formal solution of radiative transfer equation
intens = np.trapz(
    c/wl**5/(np.exp(hck/wl/tgas.reshape((-1,nz)))-1.0) * np.exp(-tau),
    -tau
).reshape(ny,nx,nt) / 5e14 # arbitrary normalization to make mean roughly 1

# save intensity images as FITS files and figures in PNG format
if (not os.path.exists(image_dir)):
    os.makedirs(image_dir)
for i in range(0, nt):
    hdu = fits.PrimaryHDU(intens[:,:,i])
    hdu.writeto(image_dir + '/image' + str(i) + '.fits', overwrite = True)
    plot.image(intens[:,:,i], image_dir + '/image'+str(i), save = True)
