# figure comparing images from BiFrost and from simulations
# vertical slices for (y,x,t,z) axes

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales90 import scales
from saha_tf import saha
import matplotlib.pyplot as plt
from astropy.io import fits
import math
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'fig_images90',
    description = 'Figure comparing Bifrost and PINN images'
)
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'neural network model')
parser.add_argument('-o', '--output_figure',
                    type = str,
                    default = 'fig_images.pdf',
                    help = 'figure file name')
parser.add_argument('-l', '--label',
                    type = str,
                    default = 'PINN',
                    help = 'label to use for neural network')

args = parser.parse_args()
model_name =  args.input_model     # name of PINN model to read
output_file = args.output_figure  # name of output figure file
label_nn =    args.label

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params
input_scale, input_offset, output_scale, output_offset = scales()

# read images
imag = np.zeros((nt,ny,nx), dtype=np.float32)
for it in range(0, nt):
    imag[it,:,:] = fits.getdata('imag_bifrost/image'+str(it)+'.fits')

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# read NN model
model = tf.keras.models.load_model(model_name, compile=False)

# load continuum opacity neural network
contopac = tf.keras.models.load_model('Opacity/contopac500.keras', compile=False)

Y = model.predict(X, batch_size = nx*ny*nz)
vx = Y[:,0]
vy = Y[:,1]
vz = Y[:,2]
lr = Y[:,3]
lt = Y[:,4]
tgas = np.power(10.0, lt)
rho =  np.power(10.0, lr)

# hydrogen ionization fraction
ionfrac = saha(tgas, rho)

# gas pressure EOS
lnp = (lr + lt) * math.log(10) + 8.7711096 + np.log(1.0 + ionfrac * 0.934)
pgas = np.exp(lnp)

# continuum opacity; contopac works in CGS units
kappa = np.power(10.0, contopac(np.stack((lt, lnp / math.log(10) + 1), axis=1))[:,0] - 1)

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

# optical depth integrating from top to bottom in z-axis using trapezoid rule
tau = cumtrapz(
    np.reshape(kappa*rho, (ny,nx,nt,nz))[:,:,:,::-1], 
    -zaxis[::-1], 
    initial = 1e-7
)[:,:,:,::-1]

c =   1.191066E-5 # 2*h*c^2
hck = 1.438832334 # h*c/k
wl =  500e-7 # wavelength in cm
# formal solution of RTE
intens = np.trapz(
    c/wl**5/(np.exp(hck/wl/tgas.reshape((ny,nx,nt,nz)))-1.0) * np.exp(-tau),
    -tau
) / 5e14 # arbitrary normalization to make mean roughly 1


# determine minimum and maximum values
# for i in range(5):
#     mins[i] = np.array(
#         Y[:,i].reshape((nt,ny,nx,nz))[t_plot,:,:,z_plot].min(), 
#         par[t_plot,z_plot,:,:,i].min()).min()
#     maxs[i] = np.array(
#         Y[:,i].reshape((nt,ny,nx,nz))[t_plot,:,:,z_plot].max(), 
#         par[t_plot,z_plot,:,:,i].max()).max()

# figure size in inches
fig = plt.figure(figsize=(10,4))

nrow = 2
ncol = 5

par_title = ['0s', '200s', '400s', '600s', '800s']
# par_cmap  = ['PiYG', 'PiYG', 'seismic', 'inferno', 'afmhot']

for i in range(5):
    fb = fig.add_subplot(nrow, ncol, i + 1)
    fb.imshow(imag[i*4], # aspect = aspect, 
        interpolation='bicubic', #vmin = mins[i], vmax = maxs[i], 
        cmap='inferno',
        extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,yaxis[0]/1000.0,yaxis[-1]/1000.0))
    if (i==0):
        plt.ylabel('BiFrost')
    plt.gca().set_yticklabels([])
    plt.gca().set_xticklabels([])
    plt.title(par_title[i])

    fn = fig.add_subplot(nrow, ncol, i + 1 + npar)
    fn.imshow(intens[:,:,i*4], # aspect = aspect, 
        interpolation='bicubic', #vmin = mins[i], vmax = maxs[i], 
        cmap='inferno',
        extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,yaxis[0]/1000.0,yaxis[-1]/1000.0))
    if (i==0):
        plt.ylabel(label_nn)
    plt.gca().set_yticklabels([])
    plt.xlabel('km')

# plt.show()
plt.savefig(output_file, format="pdf", bbox_inches="tight")

