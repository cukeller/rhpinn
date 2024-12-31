# figures of PDE terms as a function of height

import extract60 as extr
from readBiFrost60 import readBiFrost
from saha_tf import saha

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# === adjustable parameters ===
model_name = 'bifnn.keras'


# load BiFrost model
params = ['ux', 'uy', 'uz', 'lgr', 'lgp', 'lgtg']
apar, (xaxis, yaxis, zaxis, taxis), sc = readBiFrost(params, extr, 1, False)
npar = len(params) # number of physical quantities to fit
[nt, nz, ny, nx, npar] = apar.shape # number of grid points in each axis
ndim = apar.ndim - 1 # number of dimensions; par also has an axis for params
# apar = np.moveaxis(apar,1,3)

zdiff = np.zeros(len(zaxis), dtype="float32")
zdiff[:nz-1] = zaxis[1:] - zaxis[:nz-1]

# load PINN model
model = tf.keras.models.load_model(model_name, compile = False)
# generate physics parameters from PINN model at BiFrost grid locations
tm, ym, xm, zm = np.meshgrid(taxis, yaxis, xaxis, zaxis, indexing = 'ij')
X = tf.convert_to_tensor(
    np.stack((tm, ym, xm, zm), axis = -1).reshape(-1, ndim),
    dtype="float32"
)

# === constants ===
ndim = 4 # number of dimensions (x,y,z,t)
gsun = 274.0 # solar surface gravity in m/s^2
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4
eVtoJ = 1.602176634e-19 # 1eV in Joule
amu = 1.660539e-27 # atomic mass unit
mmw = 1.29 # mean molecular weight of neutral solar composition gas in amu

def integrate(y, x, axis=-1, reverse=False):
    length = y.shape[axis]
    if (reverse):
        index0 = tf.range(length - 1, 0, -1)
        index1 = tf.range(length, 1, -1)
        dx = (
            tf.gather(x, index0, axis=axis) -
            tf.gather(x, index1, axis=axis)
        )
    else:
        index0 = tf.range(0, length - 1)
        index1 = tf.range(1, length)
        dx = (
            tf.gather(x, index1, axis=axis) -
            tf.gather(x, index0, axis=axis)
        )

    return 0.5 * tf.reduce_sum(
        (tf.gather(y, index1, axis=axis) +
        tf.gather(y, index0, axis=axis)) * dx,
        axis=axis
    )


# load Rosseland neural network
rosseland = tf.keras.models.load_model('opann.keras', compile=False)

# calculate PDE terms
@tf.function
def loss_physics(t, y, x, z, model, nz, zdiff):
    '''coordinates are in physical units, i.e. meters'''

    with tf.GradientTape(persistent = True) as g:
        g.watch([x,y,z,t])

        Xf = tf.reshape(tf.stack([t, y, x, z], axis = -1),[-1, ndim])
        Y = model(Xf)

        # extract estimated physical quantities
        # DNN uses 10log, but PDEs work with ln, >>> could change that
        vx =    Y[:,0]
        vy =    Y[:,1]
        vz =    Y[:,2]
        lnrho = Y[:,3] * math.log(10)
        lt =    Y[:,4]

        rho = tf.exp(lnrho)

        # hydrogen ionization fraction
        # might not need to track gradient on this ???
        ionfrac = saha(tf.pow(10.0,lt), rho)

        # ideal gas law for solar photosphere mean molecular weight
        lnp = lnrho + lt * math.log(10) + 8.7711096 + tf.math.log(1.0 + ionfrac * 0.934)

        # --->>> might be able to express almost everything in terms of ln()
        pgas = tf.exp(lnp)

        # internal energy of gas
        e = 3.0/2.0 * pgas / rho + ionfrac * 0.934 / (amu * 1.29) * 13.6 * eVtoJ

    # these lines can be outside of GradientTape as no derivatives are involved

    # opacity, rosseland() is in CGS units
    # >>> check whether we need gas pressure or total pressure
    kappa = tf.pow(10.0, rosseland(tf.stack((lt, lnp / math.log(10) + 1), axis=1))[:,0] - 1)

    # calculate derivatives
    [dvx_dx, dvx_dy, dvx_dz, dvx_dt] = g.gradient(vx,    [x, y, z, t])
    [dvy_dx, dvy_dy, dvy_dz, dvy_dt] = g.gradient(vy,    [x, y, z, t])
    [dvz_dx, dvz_dy, dvz_dz, dvz_dt] = g.gradient(vz,    [x, y, z, t])
    [dlr_dx, dlr_dy, dlr_dz, dlr_dt] = g.gradient(lnrho, [x, y, z, t])
    [dlp_dx, dlp_dy, dlp_dz]         = g.gradient(lnp,   [x, y, z])
    [de_dx,  de_dy,  de_dz,  de_dt]  = g.gradient(e,     [x, y, z, t])

    # continuity equation
    lc0 = tf.reduce_mean(tf.reshape(tf.abs(dlr_dt),   [-1,nz]), axis=[0]) 
    lc1 = tf.reduce_mean(tf.reshape(tf.abs(vx*dlr_dx),[-1,nz]), axis=[0]) 
    lc2 = tf.reduce_mean(tf.reshape(tf.abs(vy*dlr_dy),[-1,nz]), axis=[0]) 
    lc3 = tf.reduce_mean(tf.reshape(tf.abs(vz*dlr_dz),[-1,nz]), axis=[0]) 
    lc4 = tf.reduce_mean(tf.reshape(tf.abs(dvx_dx),   [-1,nz]), axis=[0]) 
    lc5 = tf.reduce_mean(tf.reshape(tf.abs(dvy_dy),   [-1,nz]), axis=[0]) 
    lc6 = tf.reduce_mean(tf.reshape(tf.abs(dvz_dz),   [-1,nz]), axis=[0])

    # momentum equations in x, y and z
    lx0 = tf.reduce_mean(tf.reshape(tf.abs(dvx_dt),      [-1,nz]), axis=[0]) 
    lx1 = tf.reduce_mean(tf.reshape(tf.abs(vx*dvx_dx),   [-1,nz]), axis=[0]) 
    lx2 = tf.reduce_mean(tf.reshape(tf.abs(vy*dvx_dy),   [-1,nz]), axis=[0]) 
    lx3 = tf.reduce_mean(tf.reshape(tf.abs(vz*dvx_dz),   [-1,nz]), axis=[0]) 
    lx4 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho * dlp_dx),   [-1,nz]), axis=[0]) 

    ly0 = tf.reduce_mean(tf.reshape(tf.abs(dvy_dt),      [-1,nz]), axis=[0]) 
    ly1 = tf.reduce_mean(tf.reshape(tf.abs(vx*dvy_dx),   [-1,nz]), axis=[0]) 
    ly2 = tf.reduce_mean(tf.reshape(tf.abs(vy*dvy_dy),   [-1,nz]), axis=[0]) 
    ly3 = tf.reduce_mean(tf.reshape(tf.abs(vz*dvy_dz),   [-1,nz]), axis=[0]) 
    ly4 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho * dlp_dy),   [-1,nz]), axis=[0]) 

    lz0 = tf.reduce_mean(tf.reshape(tf.abs(dvz_dt),      [-1,nz]), axis=[0]) 
    lz1 = tf.reduce_mean(tf.reshape(tf.abs(vx*dvz_dx),   [-1,nz]), axis=[0]) 
    lz2 = tf.reduce_mean(tf.reshape(tf.abs(vy*dvz_dy),   [-1,nz]), axis=[0]) 
    lz3 = tf.reduce_mean(tf.reshape(tf.abs(vz*dvz_dz),   [-1,nz]), axis=[0]) 
    lz4 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho * dlp_dz + gsun),   [-1,nz]), axis=[0])

    # lmx = tf.reduce_mean(tf.square(
    #     dvx_dt + vx*dvx_dx + vy*dvx_dy + vz*dvx_dz + pgas/rho * dlp_dx
    # )) / 1.0e6 # tf.math.reduce_variance(vx)

    # lmy = tf.reduce_mean(tf.square(
    #     dvy_dt + vx*dvy_dx + vy*dvy_dy + vz*dvy_dz + pgas/rho * dlp_dy
    # )) / 1.0e6 # tf.math.reduce_variance(vy)

    # lmz = tf.reduce_mean(tf.square(
    #     # + gsun because gravity vector goes in negative z direction
    #     dvz_dt + vx*dvz_dx + vy*dvz_dy + vz*dvz_dz + pgas/rho * dlp_dz + gsun
    # )) / tf.math.reduce_variance(vz)

    # optical depth using box integration, --->>> should use trapeze; easy to do
    tau = tf.cumsum(
        tf.reshape(kappa*rho, [-1, nz]) * zdiff,
        axis=1,
        reverse=True
    )

    tau1d = tf.reshape(tau,[-1]) # flatten
    # new qrad that is per unit mass
    qrad = (0.3842494 * sigma * kappa * tf.pow(10.0, 4.0 * lt) * 
            tf.exp(-tau1d * 0.0095411586) / (1.0 + tau1d * 6.4123835))

    # energy equation
    le0 = tf.reduce_mean(tf.reshape(tf.abs(de_dt),             [-1,nz]), axis=[0])
    le1 = tf.reduce_mean(tf.reshape(tf.abs(vx*de_dx),          [-1,nz]), axis=[0]) 
    le2 = tf.reduce_mean(tf.reshape(tf.abs(vy*de_dy),          [-1,nz]), axis=[0]) 
    le3 = tf.reduce_mean(tf.reshape(tf.abs(vz*de_dz),          [-1,nz]), axis=[0]) 
    le4 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho*dvx_dx),   [-1,nz]), axis=[0]) 
    le5 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho*dvy_dy),   [-1,nz]), axis=[0]) 
    le6 = tf.reduce_mean(tf.reshape(tf.abs(pgas/rho*dvz_dz),   [-1,nz]), axis=[0])
    le7 = tf.reduce_mean(tf.reshape(tf.abs(qrad),              [-1,nz]), axis=[0])

    return tf.stack([
        lc0,lc1,lc2,lc3,lc4,lc5,lc6,
        lx0,lx1,lx2,lx3,lx4,
        ly0,ly1,ly2,ly3,ly4,
        lz0,lz1,lz2,lz3,lz4,
        le0,le1,le2,le3,le4,le5,le6,le7
        ])

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '.' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# legends for PDE terms
term_label = [
    ['continuity', 
        '$\\partial\\ln\\rho / \\partial t$', 
        '$v_x\\partial\\ln\\rho / \\partial x$',
        '$v_y\\partial\\ln\\rho / \\partial y$', 
        '$v_z\\partial\\ln\\rho / \\partial z$', 
        '$\\partial v_x / \\partial x$', 
        '$\\partial v_y / \\partial y$', 
        '$\\partial v_z / \\partial z$'
    ],
    ['momentum_x',
        '$\\partial v_x / \\partial t$', 
        '$v_x\\partial v_x / \\partial x$', 
        '$v_y\\partial v_x / \\partial y$', 
        '$v_z\\partial v_x / \\partial z$', 
        '$p/\\rho \\partial\\ln\\rho / \\partial x$'
    ],
    ['momentum_y',
        '$\\partial v_y / \\partial t$', 
        '$v_x\\partial v_y / \\partial x$', 
        '$v_y\\partial v_y / \\partial y$', 
        '$v_z\\partial v_y / \\partial z$', 
        '$p/\\rho \\partial\\ln\\rho / \\partial y$'
    ],
    ['momentum_z',
        '$\\partial v_z / \\partial t$', 
        '$v_x\\partial v_z / \\partial x$', 
        '$v_y\\partial v_z / \\partial y$', 
        '$v_z\\partial v_z / \\partial z$', 
        '$p/\\rho \\partial\\ln\\rho / \\partial z+g_{sun}$'
    ],
    ['energy',
        '$\\partial e / \\partial t$', 
        '$v_x\\partial e / \\partial x$', 
        '$v_y\\partial e / \\partial y$', 
        '$v_z\\partial e / \\partial z$', 
        '$p/\\rho \\partial v_x / \\partial x$', 
        '$p/\\rho \\partial v_y / \\partial y$', 
        '$p/\\rho \\partial v_z / \\partial z$', 
        '$q_{rad}$'
    ]
]

line_style = [
    [':', ':', ':', '-.', '--', '--', '-'],
    ['-.', '--', ':', '-.', '-'],
    ['-.', ':', '--', '-.', '-'],
    ['-.', ':', ':', '--', '-'],
    ['--', ':', ':', '-', '-.','-.', '-.', '--']
]
# calculate physics losses in batches; important for machines with little RAM
bs = nx * nz * nt * 4 # batch size
nb = (nx*ny*nz*nt) // bs # number of batches
loss = np.zeros((7+3*5+8,nz))

for ib in range (0, nb):
    istart = bs * ib
    iend   = bs * (ib + 1)
    lossarray = loss_physics(
        X[istart:iend,0], X[istart:iend,1], X[istart:iend,2], X[istart:iend,3],
        model, nz, zdiff
    )
    print_progress_bar(ib, nb, fill='=')
    loss = loss + lossarray.numpy()

loss = loss / nb

# plt.rc('font', family='Times New Roman')

nl = 0 # counter as loss array index
for j in range(len(term_label)):
    tlabel = term_label[j]
    lstyle = line_style[j]
    for i in range (len(tlabel) - 1):
        plt.plot(zaxis/1e3, loss[nl,:], label = tlabel[i+1], ls=lstyle[i])
        nl = nl + 1
    plt.ylabel('mean absolute value of terms')
    plt.legend(prop={'family':'Times New Roman'})
    plt.xlabel('z[km]')
    # plt.title(tlabel[0])
    plt.savefig('fig_' + tlabel[0] + '.pdf', format="pdf", bbox_inches="tight")
    plt.clf()
    # plt.show()
