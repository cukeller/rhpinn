# figure comparing directly trained neural network and BiFrost simulations
# vertical slices for (y,x,t,z) axes
# - shows tau=1 line for slice
# --->>> need to interpolate height where tau=1

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales90 import scales
from saha_tf import saha
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'fig_bifpinnv90b',
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
                    help = 'z-axis offset in km to make z=0 equal to tau=1 in Bifrost simulations')
parser.add_argument('-zn', '--zoffset_nn',
                    type = float,
                    default = 0.0,
                    help = 'z-axis offset in km to make z=0 equal to tau=1 in neural network')

args = parser.parse_args()

# === adjustable parameters ===
model_name =  args.input_model    # name of PINN model to read
output_file = args.output_figure  # name of output figure file
label_nn =    args.label          # label for plot of neural network
zb =          args.zoffset_bifrost
zn =          args.zoffset_nn

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis0, taxis), sc = readBiFrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params

# zaxis is not uniform, double points and make uniform for display of Bifrost and PINN
zmin = np.amin(zaxis0)
zmax = np.amax(zaxis0)
zaxis = np.arange(zmin,zmax,(zmax-zmin)/(2*nz))
nz = nz*2

# plot cuts in y and t
y_plot = ny//2
t_plot = nt//2

# Bifrost z-axis is not regular, interpolate in z for y,t cut
par_yt = np.zeros((nz,nx,npar),dtype=np.float32)
for ix in range(nx):
    for ipar in range(npar):
        par_yt[:,ix,ipar] = np.interp(zaxis,zaxis0,par[t_plot,:,y_plot,ix,ipar])


# make grid where PINN is evaluated
ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# read NN model and predict physical parameters
model = tf.keras.models.load_model(model_name, compile=False)

mod = model.predict(X, batch_size = nx*ny*nz).reshape(ny,nx,nt,nz,npar + 4)

# optical depth calculation

# constants
eVtoJ = 1.602176634e-19 # 1eV in Joule
amu = 1.660539e-27 # atomic mass unit
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4

rho =  np.power(10.0, mod[:,:,:,:,3])
temp = np.power(10.0, mod[:,:,:,:,4])

# hydrogen ionization fraction
ionfrac = saha(temp, rho)
# ideal gas law for solar photosphere mean molecular weight
pgas = np.exp(np.log(rho) + np.log(temp) + 8.7711096 + np.log(1.0 + ionfrac * 0.934))

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

# find minimum of parabola
def paramin(x1,x2,x3,y1,y2,y3):
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    a = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    b = (x3**2 * (y1-y2) + x2**2 * (y3-y1) + x1**2 * (y2-y3)) / denom
    return -b/2/a

# load continuum opacity, opacity is in CGS units
rosseland = tf.keras.models.load_model('Opacity/contopac500.keras', compile=False)

# optical depth for PINN
kappa = np.power(10.0, 
    rosseland(
        np.stack((
            mod[:,:,:,:,4].flatten(),
            np.log(pgas.flatten()) / math.log(10) + 1.0), 
            axis=1)
    ).numpy()[:,0] 
    - 1)

# plot opacity
# plt.plot(
#     zaxis/1e3,
#     np.log10(np.mean(kappa.reshape(-1,nz)*rho.reshape(-1,nz), axis=0))
# )
# plt.xlabel('z[km]')
# plt.ylabel('log(kappa*rho)')
# plt.title('opacity')
# plt.show()

# optical depth; integrating from top to bottom in z-axis using trapezoid rule
tau = cumtrapz(
    (kappa.reshape(-1,nz)*rho.reshape(-1,nz))[:,::-1], 
    -zaxis[::-1], 
    initial = 1e-7
)[:,::-1]

# plt.plot(zaxis/1e3, np.log10(np.mean(tau, axis=0)))
# plt.xlabel('z[km]')
# plt.ylabel('log(tau)')
# plt.title('optical depth')
# plt.show()

# find optical depth unity in slice
tau1line = np.argmin(np.square(tau.reshape([ny,nx,nt,nz])[y_plot,:,t_plot,:] - 1.0), axis=1)
print('z(tau=1) for PINN', np.mean(zaxis[tau1line])/1e3)

# plt.imshow(np.square(tau.reshape([ny,nx,nt,nz])[y_plot,:,t_plot,:] - 1.0))
# plt.imshow(np.rot90(np.log10(tau.reshape([ny,nx,nt,nz])[y_plot,:,t_plot,:])))
# plt.imshow(np.log10(tau.reshape([ny,nx,nt,nz])[y_plot,:,t_plot,:]))
# plt.plot(tau1line, np.arange(nx), color="grey")
# plt.show()

# print(np.mean(tau1line)) # should be around 75
# print(np.std(zaxis[tau1line])/1e3) # should be around 30km

# optical depth for Bifrost simulations
rho_b =  np.power(10.0, par_yt[:,:,3])
temp_b = np.power(10.0, par_yt[:,:,4])
ionfrac_b = saha(temp_b, rho_b)
pgas_b = np.exp(np.log(rho_b) + 
                np.log(temp_b) + 
                8.7711096 + 
                np.log(1.0 + ionfrac_b * 0.934)
    )
# plt.plot(np.mean(rho[y_plot,:,t_plot,:], axis=0))
# plt.plot(np.mean(rho_b, axis=1))
# plt.show()

# plt.plot(np.mean(temp[y_plot,:,t_plot,:], axis=0))
# plt.plot(np.mean(temp_b, axis=1))
# plt.show()

# plt.plot(np.mean(pgas[y_plot,:,t_plot,:], axis=0))
# plt.plot(np.mean(pgas_b, axis=1))
# plt.show()

kappa_b = np.power(10.0, 
    rosseland(
        np.stack((
            par_yt[...,4].flatten(),
            np.log(pgas_b.flatten()) / math.log(10) + 1.0), 
            axis=1)
    ).numpy()[:,0] 
    - 1)
tau_b = cumtrapz(
    (kappa_b.reshape(nz,nx)*np.power(10.0,par_yt[...,3]))[::-1,:], 
    -zaxis[::-1],
    axis = 0,
    initial = 1e-7
)[::-1,:]
tau1line_b = np.argmin(np.square(tau_b - 1.0), axis=0)
# print(tau1line_b)
print('z(tau=1) for Bifrost', np.mean(zaxis[tau1line_b])/1e3)

# plt.imshow(np.log10(tau_b))
# plt.plot(tau1line_b, np.arange(nx), color="grey")
# plt.show()

mins = np.zeros(npar)
maxs = np.zeros(npar)

# determine minimum and maximum values
for i in range(npar):
    mins[i] = np.array(
        mod[y_plot,:,t_plot,:,i].min(), 
        par_yt[...,i].min()).min()
    maxs[i] = np.array(
        mod[y_plot,:,t_plot,:,i].max(), 
        par_yt[...,i].max()).max()

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
    fb.imshow(par_yt[::-1,:,i], aspect = 'auto',
        interpolation='bicubic', 
        vmin = mins[i], vmax = maxs[i], 
        cmap=par_cmap[i],
        extent=extent_b)
    fb.plot(xaxis/1000.0, zaxis[tau1line_b]/1000.0+zb, color="grey")
    if (i==0):
        plt.title('BiFrost')
    plt.ylabel('km')
    if (i < npar-1):
        plt.gca().set_xticklabels([])
    else:
        plt.xlabel('km')

    # PINN
    fn = fig.add_subplot(nrow, ncol, (i + 1) * 2)
    fn.imshow(np.rot90(mod[y_plot,:,t_plot,:,i]), aspect = 'auto', 
        interpolation='bicubic', 
        vmin = mins[i], vmax = maxs[i], 
        cmap = par_cmap[i],
        extent=extent_n)
    fn.plot(xaxis/1000.0, zaxis[tau1line]/1000.0+zn, color="grey")
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

