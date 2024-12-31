# figure of radiative loss from approximation and 'missing' energy in BiFROST
# - compares energy, not energy per mass

import extract60 as extr
from readBiFrost60 import readBiFrost
from scales60 import scales
from saha_tf import saha
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# === adjustable parameters ===
step = 2
model_name = 'bifnn.keras' # name of PINN model to read

# === constants ===
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4
eVtoJ = 1.602176634e-19 # 1eV in Joule
amu = 1.660539e-27 # atomic mass unit in kg
mmw = 1.29 # mean molecular weight of neutral solar composition gas in amu

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1, False)
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions

input_scale, input_offset, output_scale, output_offset = scales()

xaxis = xaxis[::step]
nx = len(xaxis)
yaxis = yaxis[::step]
ny = len(yaxis)
taxis = taxis[::step]
nt = len(taxis)

# double number of points in z direction, use regular spacing
zaxis = zaxis[21:]
nz = len(zaxis)
zmin = np.amin(zaxis)
zmax = np.amax(zaxis)
zaxis = np.arange(zmin,zmax,(zmax-zmin)/(2*nz), dtype=np.float32)
nz = len(zaxis)

# coordinates of BiFrost simulation points
tm, ym, xm, zm = np.meshgrid(taxis, yaxis, xaxis, zaxis, indexing='ij')
X = tf.convert_to_tensor(
    np.stack((tm, ym, xm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# load Rosseland opacity neural network
rosseland = tf.keras.models.load_model('opann.keras', compile=False)

# partial physics to determine missing internal energy
def physics(t, y, x, z):

    with tf.GradientTape(persistent=True) as g:
        g.watch([x,y,z,t])

        Xf = tf.reshape(tf.stack((t, y, x, z), axis = -1),[-1, ndim])
        Y = model(Xf)

        # estimated physical quantities
        vx =    Y[:,0] 
        vy =    Y[:,1] 
        vz =    Y[:,2] 
        lnrho = Y[:,3] * math.log(10)
        ltemp = Y[:,4]

        tgas = tf.pow(10.0, ltemp)
        rho = tf.exp(lnrho)

        # hydrogen ionization fraction
        ionfrac = saha(tgas, rho)

        lnp = lnrho + ltemp * math.log(10) + 8.7711096 + tf.math.log(1.0 + ionfrac * 0.934)
        pgas = tf.exp(lnp)

        # internal energy of partially ionized ideal gas per mass (not per volume!)
        e = 3.0/2.0 * pgas / rho + ionfrac * 0.934 / (amu * 1.29) * 13.6 * eVtoJ

    dvx_dx = g.gradient(vx,x)
    dvy_dy = g.gradient(vy,y)
    dvz_dz = g.gradient(vz,z)
    [de_dx, de_dy, de_dz, de_dt]  = g.gradient(e, [x, y, z, t])

    # non-radiative part of conservation of energy equation (per mass)
    menergy = de_dt + vx*de_dx + vy*de_dy + vz*de_dz + pgas/rho * (dvx_dx + dvy_dy + dvz_dz)

    # return missing energy per unit mass, density, log10(pressure), log10(temperature)
    return -menergy, rho, lnp / math.log(10), ltemp


# # load BiFrost-trained neural network
model = tf.keras.models.load_model(model_name, compile=False)

# determine relevant quantities over whole space
qrad, rho, lp, lt = physics(X[:,0],X[:,1],X[:,2],X[:,3])

# opacity per unit mass; rosseland() works with CGS units, not SI as the rest here
kappa = tf.pow(10.0, rosseland(tf.stack((lt, lp + 1), axis = -1))[:,0] - 1)

# general integration
def integrate(y, x, axis=-1, reverse=False, cummulative=False):
    length = y.shape[axis]
    if (cummulative):
        dx = x[1:] - x[:-1]
        # tf.print('dx', dx)
        if (reverse):
            res = tf.concat([
                0.5 * tf.cumsum(
                    (y[..., :-1] + y[..., 1:]) * dx,
                    axis=axis,
                    reverse = True
                ),
                tf.zeros_like(y[..., :1]) # only works if we integrate over last axis
            ], axis=axis)
            # tf.print(res[0,:])
            return res
        else:
            return tf.concat([
                tf.zeros_like(y[..., :1]), # only works if we integrate over last axis
                0.5 * tf.cumsum(
                    (y[..., :-1] + y[..., 1:]) * dx,
                    axis=axis
                )       
            ], axis=axis)
    else:
        if (reverse):
            index0 = tf.range(length - 1, 0, -1)
            index1 = tf.range(length, 1, -1)
            dx = tf.gather(x, index0, axis=axis) - tf.gather(x, index1, axis=axis)
        else:
            index0 = tf.range(0, length - 1)
            index1 = tf.range(1, length)
        dx = tf.gather(x, index1, axis=axis) - tf.gather(x, index0, axis=axis)
        return 0.5 * tf.reduce_sum(
            (tf.gather(y, index1, axis=axis) + tf.gather(y, index0, axis=axis)) * dx,
            axis=axis
        )

# trapezoid integration 
tau = integrate(
    tf.reshape(kappa*rho, [-1, nz]),
    zaxis,
    reverse=True,
    cummulative=True
)

meantau = np.log(np.mean(tau.numpy(),axis=0))

tau1d = tf.reshape(tau,[-1])

qrad = qrad * rho

# fitted parameters of radiative loss approximation
a = 0.338290532430013
b = 0.01970027565956116
c = 4.92561757564544

# radiative loss approximation
qfrad = a*sigma*kappa * rho * tf.pow(10.0, 4.0 * lt)*tf.exp(-tau1d * b) / (1.0 + tau1d * c)

mener = tf.reduce_mean(integrate(
    tf.reshape(qrad, [-1, nz]),
    zaxis
))
rener = tf.reduce_mean(integrate(
    tf.reshape(qfrad, [-1, nz]),
    zaxis
))
print('total missing energy in 1e7 J/m^2   ', float(mener/1e7))
print('output radiative energy in 1e7 J/m^2', float(rener/1e7))

t_plot = nt//4
y_plot = ny//2

fq =  qrad.numpy().reshape((nt, ny, nx, nz))[t_plot,y_plot,:,:]
ff = qfrad.numpy().reshape((nt, ny, nx, nz))[t_plot,y_plot,:,:]
# print(np.amin(fq), np.amax(fq), np.amin(ff), np.amax(ff))

# figure out tau500=1 and correct height in slice that is shown
# continuum opacity at 500nm
k500 = tf.pow(10.0, 
              tf.keras.models.load_model('Opacity/contopac500.keras', compile=False)(
                  tf.stack((lt, lp + 1), axis = -1))[:,0] - 1)
t500 = integrate(
    tf.reshape(k500*rho, [nt, -1, nz])[t_plot,:,:],
    zaxis,
    reverse=True,
    cummulative=True
)
t5 = tf.math.log(tf.reduce_mean(t500, axis=0)).numpy()
z0 = zaxis[np.argmin(np.abs(t5-1))]/1000.0
print('tau=1 height', z0)

# fig = plt.figure(figsize=(8,4))

fig, axs = plt.subplots(1, 2, figsize=(10,5))
# fig.set_size_inches(10, 5)

axs[0].imshow(np.rot90(fq[:,:60]),
    interpolation='bicubic', aspect='auto', vmin=-500, vmax=2300, cmap='inferno',
    extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,zaxis[0]/1000.0-z0,zaxis[60]/1000.0-z0))
axs[0].set_xlabel('km')
axs[0].set_ylabel('height [km]')
axs[0].set_title('Bifrost missing internal energy')

map = axs[1].imshow(np.rot90(ff[:,:60]),
    interpolation='bicubic', aspect='auto', vmin=-500, vmax=2300, cmap='inferno',
    extent=(xaxis[0]/1000.0,xaxis[-1]/1000.0,zaxis[0]/1000.0-z0,zaxis[60]/1000.0-z0))
axs[1].set_xlabel('km')
axs[1].set_yticklabels([])
axs[1].set_title('approximated radiative loss')

cbar = fig.colorbar(map, ax=axs, orientation='horizontal', aspect=50)
cbar.ax.set_xlabel('radiative loss [W/m$^3$]', rotation=0)

# plt.show()
plt.savefig('fig_qrad.pdf', format="pdf", bbox_inches="tight")
plt.clf()

fig = plt.figure(figsize=(6.4,4.8))
plt.plot(zaxis*1e-3-z0, np.mean( qrad.numpy().reshape((-1, nz)), axis=0), 
         color='blue',
         label='Bifrost missing internal energy'
)
plt.plot(zaxis*1e-3-z0, np.mean(qfrad.numpy().reshape((-1, nz)), axis=0), 
         color='red', 
         label = 'radiative loss approximation'
)
plt.legend()
plt.xlabel('height [km]')
plt.ylabel('radiative loss [W/m$^3$]')
# plt.show()
plt.savefig('fig_qenergy.pdf', format="pdf", bbox_inches="tight")