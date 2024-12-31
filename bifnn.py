# train deep neural network directly on Bifrost simulations
# author: Christoph U.Keller, ckeller@nso.edu

import extract
from readBifrost import readBifrost
from scales import scales
import dnn
import plot
import argparse
import numpy as np
import tensorflow as tf

# parse arguments
parser = argparse.ArgumentParser(
    prog = 'bifnn',
    description = 'Train a deep neural network to reproduce a Bifrost simulation'
)
parser.add_argument('-y', '--deltaY',
                    type = int,
                    default = 0,
                    help = 'change in y extraction')
parser.add_argument('-s', '--step',
                    type = int,
                    default = 1,
                    help = 'step when comparing to data in x,y,z')
parser.add_argument('-e', '--epochs',
                    type = int,
                    default = 100,
                    help = 'number of epochs')
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'model to start from')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'bifnn.keras',
                    help = 'neural network model name')
parser.add_argument('-l', '--learning_rate',
                    type = float,
                    default = 0.005,
                    help = 'learning rate')
parser.add_argument('-b', '--batch_size',
                    type = int,
                    default = 9216,
                    help = 'batch size')
args = parser.parse_args()


# === adjustable parameters ===
step =             args.step          # step when comparing to data in x,y,z
modelout_name =    args.output_model  # file name of trained model being saved
read_model = False                  # by default, no model is read  
if (args.input_model):
    read_model =   True               # True when reading trained input model
    modelin_name = args.input_model   # name of neural network model to read
epochs =           args.epochs        # number of epochs
batch_size =       args.batch_size    # batch size
learning_rate =    args.learning_rate # learning rate for optimizer
extract.ystart =   extract.ystart + args.deltaY # add offset in y for Bifrost data extraction
extract.ystop  =   extract.ystop  + args.deltaY

params = ['ux','uy','uz','lgr','lgtg'] # Bifrost parameters to read
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extract, step)
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions

input_scale, input_offset, output_scale, output_offset = scales()

# make into nx*ny*nz*nt by npar matrix with z-axis last
# Bifrost data are in [nt,nz,ny,nx] format; move z-axis from second to last/fastest position
Yorig = np.moveaxis(par,[0,1],[2,3]).reshape(-1, npar)

# coordinates of Bifrost simulation points
ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
Xorig = np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, ndim)

# convert numpy arrays to TF tensors
X = tf.convert_to_tensor(Xorig, dtype="float32")
Y = tf.convert_to_tensor(Yorig, dtype="float32")

# loss function
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(
        tf.reduce_mean(tf.square(y_true - y_pred[:,0:5]), axis=0) / tf.square(output_scale)
    )

# neural network: input is (y,x,t,z), output is (ux, uy, uz, log(rho), log(T), y, x, t, z)
# all inputs and outputs are in physical units
model = dnn.model()

if (read_model):
    # read previous model and overwrite global model define above
    globals()['model'] = tf.keras.models.load_model(modelin_name, compile=False)

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate
    ),
    loss = loss_fn
)

# save best model
check_point = tf.keras.callbacks.ModelCheckpoint(
    modelout_name,
    monitor = 'loss',
    save_best_only = True
)

model.fit(
    X,
    Y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = 1,
    callbacks=[check_point]
)

# print neural network structure
model.summary()

# load best model that has been saved
best_model = tf.keras.models.load_model(modelout_name, compile=False)

# predicted solution over whole space
predicted = best_model.predict(Xorig, batch_size = nx*ny*nz//4, verbose = 0)
print('MAE data errors normalized to -1 to +1 range')
for i in range(npar):
    print(
        params[i],
        np.mean(np.abs(predicted[:,i] - Yorig[:,i])) / np.asarray(output_scale)[i]
    )

predicted = predicted.reshape((ny, nx, nt, nz, npar + ndim))

# coordinates for 2-D slices in neural network and Bifrost simulations
t_plot = nt // 2
z_plot = nz // 2
y_plot = ny // 2

# xy plots
for i in range(0,npar):
    plot.comparison(
        par[t_plot,z_plot,:,:,i], 
        predicted[:,:,t_plot,z_plot,i], 
        params[i]
    )

# xz_plots
for i in range(0,npar):
    plot.comparison(
        par[t_plot,:,y_plot,:,i], 
        np.transpose(predicted[y_plot,:,t_plot,:,i]), 
        params[i], 
        True
    )
