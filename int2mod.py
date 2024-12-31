# train network to make initial guesses of physical parameters uz, lr, lt
# based on normalized continuum intensity
#   - uses Bifrost simulations directly and not a DNN trained on Bifrost
#   - applies a weight in the z-axis to maximize correspondence in layers where light emerges from
# additional constraints:
#   - zero net mass motion through each horizontal layer
#   - mean vz^2 compatible with zero net mass flux through layers
#   - mean density and temperature stratifications
#   - variance of the density and temperature
#   - correlation between mass flux and temperature
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
from scales import scales
import numpy as np
import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'int2mod',
    description = 'Train a neural network to provide physical parameters based on intensity images alone'
)
parser.add_argument('-y', '--deltaY',
                    type = int,
                    default = 0,
                    help = 'change in y extraction')
parser.add_argument('-i', '--image_directory',
                    type = str,
                    help = 'image directory')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'int2mod.keras',
                    help = 'neural network output model name')
parser.add_argument('-e', '--epochs',
                    type = int,
                    default = 200,
                    help = 'number of epochs')
parser.add_argument('-l', '--learning_rate',
                    type = float,
                    default = 0.0005,
                    help = 'learning rate')
parser.add_argument('-b', '--batch_size',
                    type = int,
                    default = 49152,
                    help = 'batch size')
args = parser.parse_args()


# === adjustable parameters ===
model_name =       args.output_model  # file name of trained model being saved
epochs =           args.epochs        # number of epochs
batch_size =       args.batch_size    # batch size
learning_rate =    args.learning_rate # learning rate for optimizer
extr.ystart =      extr.ystart + args.deltaY # add offset in y for Bifrost data extraction
extr.ystop  =      extr.ystop  + args.deltaY
image_name =       args.image_directory     # directory with images

params = ['uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, 1)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
input_scale, input_offset, output_scale, output_offset = scales()

# weight in z-axis for fitting BiFrost parameters, more weight where light emerges
znorm = zaxis*input_scale[3] + input_offset[3]
zweight = np.sqrt(1.0 + np.exp(-np.power(znorm / 0.25, 2)) * 20.0)
zweight = tf.convert_to_tensor(zweight / np.mean(zweight), dtype="float32") # normalize to a mean of 1.0

# remove net mass flux from BiFrost simulations
par = np.moveaxis(par,1,3) # moves z-axis to last coordinate place
for it in range(nt):
    dens = np.power(10.0, par[it,...,1])
    par[it,...,0] = par[it,...,0] - (
        np.mean(par[it,...,0]*dens, axis=(0,1)) / np.mean(dens, axis=(0,1))
    )

# load intensity images and normalize to mean 0 and stdv 1
imag = np.zeros((nt,ny,nx), dtype=np.float32)
for it in range(nt):
    imag[it,:,:] = fits.getdata(image_name+'/image'+str(it)+'.fits')
    imag[it,:,:] = imag[it,:,:] - np.mean(imag[it,:,:])
    imag[it,:,:] = imag[it,:,:] / np.std(imag[it,:,:])
imag = tf.convert_to_tensor(imag.flatten(), dtype="float32")

ndim = 2 # number of dimensions: (flattened) intensity, z
im, zm = np.meshgrid(imag, zaxis, indexing='ij')
X_orig = np.stack((im, zm), axis = -1).reshape(-1, nz, ndim)
Y_orig = par.reshape(-1, nz, npar)

# variance of vertical velocity
vvz =   np.var( Y_orig[:,:,0], axis=0)
# mean and variance of log density
mlrho = np.mean(Y_orig[:,:,1], axis=0)
vlrho = np.var( Y_orig[:,:,1], axis=0)
# mean and variance of log temperature
mlogt = np.mean(Y_orig[:,:,2], axis=0)
vlogt = np.var( Y_orig[:,:,2], axis=0)

# correlation between temperature and mass flux
vz = Y_orig[:,:,0]
rho = np.power(10, Y_orig[:,:,1])
# --->>> this should probably be done per time step and not averaged over time!
massflux = rho * vz - np.mean(rho * vz, axis=0)
temp = np.power(10.0, Y_orig[:,:,2])
vtcorr = np.mean(
    (temp - np.mean(temp, axis=0, keepdims = True)) * massflux, 
    axis=0
)

# Keras custom sequence generator that shuffles t,x,y points for every epoch
class CustomSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        super().__init__()
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        # print('len(self.x)',len(self.x))
        p = np.random.permutation(len(self.x))
        self.x_shuffled = self.x[p,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1,npar))

    # returns number of batches
    def __len__(self):
        return int(np.ceil(len(self.x_shuffled) / float(self.batch_size)))

    # returns one batch at batch index idx
    def __getitem__(self, idx):
        batch_x = self.x_shuffled[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_shuffled[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)

    # reshuffle at end of each epoch   
    def on_epoch_end(self):
        p = np.random.permutation(len(self.x))
        self.x_shuffled = self.x[p,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1,npar))


# neural network:
#   input is normalized intensity and z-coordinate in meters
#   output is uz, log10(rho), log10(T) along z
#   all inputs and outputs are in physical coordinates
model = tf.keras.Sequential([
    tf.keras.layers.Input((ndim,)),
    tf.keras.layers.Rescaling(
        scale  = [1.0, input_scale[3]],
        offset = [0.0, input_offset[3]]
    ),
    tf.keras.layers.Dense(8, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(npar, activation='linear'),
    tf.keras.layers.Rescaling(
        # [2:5] is scaling for [uz, log10(rho), log10(T)]
        scale  = output_scale[2:5],
        offset = output_offset[2:5]
    )
])


def losses(y_true, y_pred):
    # minimizing vertical net mass flow should only control vertical velocity
    rho = tf.reshape(tf.pow(10.0, tf.stop_gradient(y_pred[:,1])),[-1,nz])
    vz = tf.reshape(y_pred[:,0],[-1,nz])
    # --->>> this should probably be done per time step and not averaged over time!
    massflux = rho * vz - tf.reduce_mean(rho * vz, axis=0, keepdims = True)
    temp = tf.reshape(tf.pow(10.0, y_pred[:,2]),[-1,nz])

    return tf.stack([
        # mean vertical mass motion in each layer should be zero
        # --->>> should be true for each time step
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,0], [-1,nz]) * rho, axis=0) /
                tf.reduce_mean(rho, axis=0)
            )
        ),

        # difference of variance of vertical velocity
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,0], [-1,nz]), axis=0) - vvz
            )
        ),

        # difference of mean of logarithm of density
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,1], [-1,nz]), axis=0) - mlrho
            )
        ),

        # difference of variance of logarithm of density
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,1], [-1,nz]), axis=0) - vlrho
            )
        ),

        # difference of mean of logarithm of temperature
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(tf.reshape(y_pred[:,2], [-1,nz]), axis=0) - mlogt
            )
        ),

        # difference of variance of logarithm of temperature
        tf.reduce_mean(
            tf.square(
                tf.math.reduce_variance(tf.reshape(y_pred[:,2], [-1,nz]), axis=0) - vlogt
            )
        ),

        # correlation between mass flux and temperature         
        tf.reduce_mean(
            tf.square(
                tf.reduce_mean(
            (temp - tf.reduce_mean(temp, axis=0, keepdims = True)) * massflux, 
            axis=0) - vtcorr
            )
        ),        

        # difference of vz
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,0] - y_pred[:,0]),[-1,nz]) * zweight) / tf.square(output_scale[2]),

        # difference of log10(rho)
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,1] - y_pred[:,1]),[-1,nz]) * zweight) / tf.square(output_scale[3]),

        # difference of log10(T)
        tf.reduce_mean(
            tf.reshape(tf.square(y_true[:,2] - y_pred[:,2]),[-1,nz]) * zweight) / tf.square(output_scale[4]),
    ])

loss_name = [
    'mean vertical mass motion           ',
    'variance of vertical velocity       ',
    'mean of logarithm of density        ',
    'variance of logarithm of density    ',
    'mean of logarithm of temperature    ',
    'variance of logarithm of temperature',
    'mass flux temperature correlation   ',
    'vz                                  ',
    'log10(rho)                          ',
    'log10(T)                            ',
]

loss_scale = [
    1e-6,  # mean vertical mass motion
    1e-13, # variance of vertical velocity
    1e-1,  # mean of logarithm of density
    1e1,   # variance of logarithm of density
    1e1,   # mean of logarithm of temperature
    1e2,   # variance of logarithm of temperature
    1e-5,  # mass flux temperature correlation
    1,     # vz
    1,     # log10(rho)
    1,     # log10(T)
]

def loss_fn(y_true, y_pred):
    return tf.reduce_sum(losses(y_true, y_pred) * loss_scale)


model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

# save best model
check_point = tf.keras.callbacks.ModelCheckpoint(
    model_name,
    monitor = 'loss',
    save_best_only = True
)

seq = CustomSequence(X_orig, Y_orig, batch_size)

model.fit(
    seq,
    epochs = epochs,
    verbose = 1,
    shuffle = False,
    callbacks=[check_point]
)

# print neural network structure
model.summary()

# load best model
best_model = tf.keras.models.load_model(model_name, compile=False)
pred = best_model.predict(X_orig.reshape(-1,ndim), batch_size=nx*nz*8).reshape((-1,nz,npar))
print(' ')
print('Mean Absolute Errors (from BiFrost) normalized to -1 to +1 range')
for i in range(len(params)):
    print(params[i], np.mean(np.abs(Y_orig[:,:,i] - pred[:,:,i]) / output_scale[i+2]))

print('losses')
l = losses(Y_orig.reshape(-1,npar), model(X_orig.reshape(-1,ndim)))
for i in range(len(l)):
    print(loss_name[i], float(l[i]*loss_scale[i]))

p = pred
d = Y_orig

for i in range(len(params)):
    plt.plot(zaxis/1e3, np.mean(np.abs(d[:,:,i] - p[:,:,i]), axis=0) / output_scale[i])
    plt.title('MAE of ' + params[i])
    plt.xlabel('z[km]')
    plt.ylabel('MAE')
    plt.show()

plt.plot(zaxis/1e3, 
    np.mean(d[:,:,0] * np.power(10,d[:,:,1]), axis=0) / 
    np.mean(np.power(10,d[:,:,1]), axis=0),
    label="BiFrost"
)
plt.plot(zaxis/1e3, 
    np.mean(p[:,:,0] * np.power(10,p[:,:,1]), axis=0) / 
    np.mean(np.power(10,p[:,:,1]), axis=0),
    label="Fit"
)
plt.title('mean vertical mass flux velocity')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('vz[m/s]')
plt.show()

plt.plot(zaxis/1e3, vvz, label="BiFrost")
plt.plot(zaxis/1e3, np.var(p[:,:,0], axis=0), label="Fit")
plt.title('variance of vertical velocity')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('(vz[m/s])^2')
plt.show()

plt.plot(zaxis/1e3, mlrho, label="BiFrost")
plt.plot(zaxis/1e3, np.mean(p[:,:,1], axis=0), label="Fit")
plt.title('mean log(density)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('rho[kg/m^3]')
plt.show()

plt.plot(zaxis/1e3, vlrho, label="BiFrost")
plt.plot(zaxis/1e3, np.var(p[:,:,1], axis=0), label="Fit")
plt.title('variance log(density)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('rho[kg/m^3]^2')
plt.show()

plt.plot(zaxis/1e3, mlogt, label="BiFrost")
plt.plot(zaxis/1e3, np.mean(p[:,:,2], axis=0), label="Fit")
plt.title('mean log(temperature)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('T[K]')
plt.show()

plt.plot(zaxis/1e3, vlogt, label="BiFrost")
plt.plot(zaxis/1e3, np.var(p[:,:,2], axis=0), label="Fit")
plt.title('variance log(temperature)')
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('T[K]^2')
plt.show()

# correlation between temperature and mass flux as a function of height
temp = np.power(10.0, p[:,:,2])
mf = p[:,:,0] * np.power(10.0,p[:,:,1]) - np.mean(p[:,:,0] * np.power(10.0,p[:,:,1]), axis=0)
mtcorr = np.mean(
    (temp - np.mean(temp, axis=0, keepdims = True)) * mf, 
    axis=0
)
plt.plot(zaxis/1e3, vtcorr, label="BiFrost")
plt.plot(zaxis/1e3, mtcorr, label="Fit")
plt.legend()
plt.xlabel('z[km]')
plt.ylabel('vz*rho-T correlation')
plt.title('vertical mass flux - temperature correlation')
plt.show()
