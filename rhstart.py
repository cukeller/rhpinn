# radiative hydro model fitting simulated intensity images
# make initial model based on intensity images alone to start the full modeling
# - this one has z as the fastest axis and 
#   vz*dlr/dz + div v = 0 and mean(vx^2), mean(vy^2) penalty
# - added argument parsing, only stores best model
# - this one has (x,y,t,z) axes order
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
from scales import scales
import dnn as dnn

import math
import argparse
import numpy as np
from astropy.io import fits
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'rhstart',
    description = 'Initial PINN training using int2mod Deep Neural Network'
)
parser.add_argument('-m', '--image_directory',
                    type = str,
                    default = 'imag_bifrost',
                    help = 'image directory')
parser.add_argument('-s', '--step',
                    type = int,
                    default = 1,
                    help = 'step when comparing to data in x,y,z')
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'model to start from')
parser.add_argument('-n', '--int2mod',
                    type = str,
                    default = 'int2mod.keras',
                    help = 'int2mod deep neural network')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'rhstart.keras',
                    help = 'neural network output model name')
parser.add_argument('-e', '--epochs',
                    type = int,
                    default = 25,
                    help = 'number of epochs')
parser.add_argument('-l', '--learning_rate',
                    type = float,
                    default = 0.002,
                    help = 'learning rate')
parser.add_argument('-b', '--batch_size',
                    type = int,
                    default = 12288,
                    help = 'batch size')
args = parser.parse_args()


# === adjustable parameters ===
step =             args.step          # step when enforcing physics in x,y,z
modelout_name =    args.output_model  # file name of trained model being saved
read_model = False                  # by default, no model is read  
if (args.input_model):
    read_model =   True               # True when reading trained input model
    modelin_name = args.input_model   # name of neural network model to read
dnn_int2mod =      args.int2mod       # 1-D DNN that uses intensity input
epochs =           args.epochs        # number of epochs
batch_size =       args.batch_size    # batch size
learning_rate =    args.learning_rate # learning rate for optimizer
image_name =       args.image_directory     # directory with images

continuity_scale = 1    # scaling of continuity equation

# read Bifrost axes
params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, step, False)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions
input_scale, input_offset, output_scale, output_offset = scales()

# read all intensity images and normalize them
imag = np.zeros((nt,ny,nx), dtype=np.float32)
for it in range(nt):
    imag[it,:,:] = fits.getdata(image_name+'/image'+str(it*step)+'.fits')[::step,::step]
    # normalize to mean 0 and stdv 1
    imag[it,:,:] = imag[it,:,:] - np.mean(imag[it,:,:])
    imag[it,:,:] = imag[it,:,:] / np.std(imag[it,:,:])

# read intensity to physical parameters network
int2mod = tf.keras.models.load_model(dnn_int2mod, compile=False)

# approximate uz, log(rho), log(T) using DNN for intensity->physics
par = np.zeros((ny,nx,nt,nz,npar), dtype=np.float32)
for it in range(0, nt):
    im, zn = np.meshgrid((imag[it,:,:]).reshape(-1), zaxis, indexing='ij')
    Xim = tf.convert_to_tensor(np.stack((im, zn), axis = -1).reshape((-1,2)), dtype="float32")
    Yim = int2mod(Xim)
    par[:,:,it,:,2] = tf.reshape(Yim[:,0],[ny,nx,nz]) # uz
    par[:,:,it,:,3] = tf.reshape(Yim[:,1],[ny,nx,nz]) # log10(rho)
    par[:,:,it,:,4] = tf.reshape(Yim[:,2],[ny,nx,nz]) # log10(T)

# make into nx*ny*nz*nt by npar matrix
f_orig = par.reshape(-1, npar)

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
Xorig = np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim)

X =      tf.convert_to_tensor(Xorig,  dtype="float32")
f_true = tf.convert_to_tensor(f_orig, dtype="float32")

# loss in continuity equation
# coordinates are in physical units, i.e. meters
def loss_physics(y, x, t, z):
    with tf.GradientTape(persistent=True) as g:
        g.watch([x,y,z])

        Xf = tf.reshape(tf.stack((y, x, t, z), axis = -1),[-1, ndim])
        Y = model(Xf)
        vx =    Y[:,0]
        vy =    Y[:,1]
        vz =    Y[:,2]
        lnrho = Y[:,3] * math.log(10)

    # calculate derivatives
    dvx_dx = g.gradient(vx,x)
    dvy_dy = g.gradient(vy,y)
    dvz_dz = g.gradient(vz,z)
    dlr_dz = g.gradient(lnrho,z)

    # continuity equation approximated by the most important terms
    # note that this only uses the derivatives of vx and vy, which means that an arbitrary
    # offset can be applied to both of them without impacting the continuity loss
    # now we also do not allow a back-propagation into lnrho and vz as we do not want to change 
    # those from what int2mod does; this one is all about getting estimates for the horizontal
    # velocities
    return (
        tf.reduce_mean(tf.square(
            tf.stop_gradient(vz * dlr_dz) + dvx_dx + dvy_dy + tf.stop_gradient(dvz_dz))) +
        tf.reduce_mean(vx**2 + vy**2) / 1e13 # was 1e13 for a long time
    )

# loss function
def loss_fn(y_true, y_pred):
    loss_continuity = loss_physics(
        y_pred[:,npar], y_pred[:,npar+1], y_pred[:,npar+2], y_pred[:,npar+3]
    )
    return (
        loss_continuity * continuity_scale +
        tf.reduce_mean(
            # do not fit vx and vy as we have no idea what they should be
            tf.reduce_mean(tf.square(y_true[:,2:5] - y_pred[:,2:5]), axis = 0) / 
            tf.square(output_scale[2:5])
        )
    )

# neural network:
#   input is (t,y,x,z)
#   output is ux, uy, uz, log10(rho), log10(T), t, y, x, z
#   all inputs and outputs are in physical units
model = dnn.model()

if (read_model):
    # read previous model and overwrite global model defined above
    globals()['model'] = tf.keras.models.load_model(modelin_name, compile=False)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

check_point = tf.keras.callbacks.ModelCheckpoint(
    modelout_name,
    monitor = 'loss',
    save_best_only = True
)

model.fit(
    X,
    f_true,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1,
    callbacks = [check_point]
)

# print neural network structure
model.summary()


# load best model
best_model = tf.keras.models.load_model(modelout_name, compile=False)
# predict solution over whole space
predicted = model.predict(Xorig, batch_size = nx*nz)
print('Mean Absolute Errors (from 1-D model) normalized to -1 to +1 range')
for i in range(2,npar):
    print(
        params[i],
        np.mean(np.abs(predicted[:,i] - f_orig[:,i])) / output_scale[i]
    )

print('Losses:')
for i in range(2,npar):
    print(
        params[i],
        np.mean((predicted[:,i] - f_orig[:,i])**2) / output_scale[i]**2
        )
loss_continuity = loss_physics(X[0], X[1], X[2], X[3])
print('continuity loss', float(loss_continuity * continuity_scale))
