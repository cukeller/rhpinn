# radiative hydro model fitting intensity images
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
from scales import scales
import dnn
from zaxis import make_zaxis
from lossScale import loss_scale
import physics_loss as lp
import argparse
import numpy as np
from astropy.io import fits
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog = 'rhmod',
    description = 'PINN training using full physics constraints'
)
parser.add_argument('-m', '--image_directory',
                    type = str,
                    help = 'image directory')
parser.add_argument('-z', '--stratification',
                    type = str,
                    default = 'strat.keras',
                    help = 'mean stratification model')
parser.add_argument('-s', '--step',
                    type = int,
                    default = 1,
                    help = 'step when comparing to data in x,y,z')
parser.add_argument('-i', '--input_model',
                    type = str,
                    help = 'model to start from')
parser.add_argument('-o', '--output_model',
                    type = str,
                    default = 'rhmod.keras',
                    help = 'neural network output model name')
parser.add_argument('-e', '--epochs',
                    type = int,
                    default = 50,
                    help = 'number of epochs')
parser.add_argument('-l', '--learning_rate',
                    type = float,
                    default = 0.0002,
                    help = 'learning rate')
parser.add_argument('-b', '--batch_size',
                    type = int,
                    default = 12800,
                    help = 'batch size')
args = parser.parse_args()

# === adjustable parameters ===
step =             args.step            # step when enforcing physics in x,y,z
modelout_name =    args.output_model    # file name of trained model being saved
read_model = False                      # by default, no model is read  
if (args.input_model):
    read_model =   True                 # True when reading trained input model
    modelin_name = args.input_model     # name of neural network model to read
epochs =           args.epochs          # number of epochs
batch_size =       args.batch_size      # batch size
learning_rate =    args.learning_rate   # learning rate for optimizer
image_name =       args.image_directory # directory with images
stratification =   args.stratification  # mean stratifications

params = ['ux','uy','uz','lgr','lgtg']
par, (xaxis, yaxis, zaxis, taxis), sc = readBifrost(params, extr, step = 1, read = False)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions; par also has an axis for params
# reduce x,y,t axes by taking into account step
xaxis = xaxis[::step]
nx = len(xaxis)
yaxis = yaxis[::step]
ny = len(yaxis)
taxis = taxis[::step]
nt = len(taxis)

zaxis = make_zaxis()
nz = len(zaxis)
zaxis_tensor = tf.convert_to_tensor(zaxis, dtype="float32")

input_scale, input_offset, output_scale, output_offset = scales()
# read images
img = np.zeros((ny,nx,nt), dtype=np.float32)
for it in range(0, nt):
    img[:,:,it] = fits.getdata(image_name+'/image'+str(it*step)+'.fits')[::step,::step]
img = np.repeat(img.reshape((-1)), nz).reshape((-1, nt, nz))

ym, xm, tm, zm = np.meshgrid(yaxis, xaxis, taxis, zaxis, indexing='ij')
Xorig = np.stack((ym, xm, tm, zm), axis = -1).reshape(-1, nt, nz, ndim)

# determine mean lgr and lgtg as a function of z using strat neural network
stratmod = tf.keras.models.load_model(stratification, compile=False)
strat = stratmod(zaxis)

# Keras custom sequence generator that shuffles x,y points for every epoch and separately in time
class CustomSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        super().__init__()
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.rng = np.random.default_rng()
        p = self.rng.permutation(self.x.shape[0])
        self.x_shuffled = self.x[p,:,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1))

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
        p = self.rng.permutation(self.x.shape[0])
        self.x_shuffled = self.x[p,:,:,:].reshape((-1,ndim))
        self.y_shuffled = self.y[p,:,:].reshape((-1))


# loss function
def loss_fn(y_true, y_pred):
    losses, intens = lp.loss_physics(
        y_pred[:,npar], y_pred[:,npar+1], y_pred[:,npar+2], y_pred[:,npar+3],
        model, nz, nt, zaxis_tensor, strat
    )
    loss_image = tf.reduce_mean(tf.square(intens - y_true[::nz]))
    return loss_scale[0] * loss_image + tf.reduce_sum(losses * loss_scale[1:])


# Deep Neural Network model
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

check_point = tf.keras.callbacks.ModelCheckpoint(
    modelout_name,
    monitor = 'loss',
    save_best_only = True
)

seq = CustomSequence(Xorig, img, batch_size)

model.fit(
    seq,
    epochs = epochs,
    shuffle = False,
    verbose = 1,
    callbacks = [check_point]
)

# print neural network structure
model.summary()
