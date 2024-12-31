# determine scale/offset for Bifrost simulation extractions
# - needed for neural network normalizing of input and output
# - writes to scales.py, which can be imported
# author: Christoph U.Keller, ckeller@nso.edu

import extract as extr
from readBifrost import readBifrost
import numpy as np

# === adjustable parameters ===
outputFileName = 'scales.py'
params = ['ux','uy','uz','lgr','lgtg']

par, (xaxis, yaxis, zaxis, taxis), (xscale, yscale, zscale, tscale) = readBifrost(params, extr)
[nt, nz, ny, nx, npar] = par.shape # number of grid points in each axis
ndim = par.ndim - 1 # number of dimensions

def scaleOffset(input):
    """
    provide offset and scaling for input such that
    output = input * scale + offset
    is in the range -1 to +1
    """
    min = np.amin(input)
    max = np.amax(input)
    scale = 2.0 / (max - min)
    offset = -min * scale - 1.0
    return (input * scale + offset, scale, offset)


# scale physical parameters to -1 to +1 range:
# velocity in x,y,z is all scaled the same with no offset
u_scale = 1.0 / np.abs(par[:,:,:,:,0:3]).max()
# density and temperature are in 10log
lr_norm, lr_scale, lr_offset = scaleOffset(par[:,:,:,:,3])
lt_norm, lt_scale, lt_offset = scaleOffset(par[:,:,:,:,4])

# coordinates in physical units (meters and seconds)
x_phys = np.arange(0, nx) * xscale
y_phys = np.arange(0, ny) * yscale
z_phys = zaxis
t_phys = np.arange(0, nt) * tscale

# determine coordinate scaling and offset to range -1 to +1
x_norm, x_scale, x_offset = scaleOffset(x_phys)
y_norm, y_scale, y_offset = scaleOffset(y_phys)
z_norm, z_scale, z_offset = scaleOffset(z_phys)
t_norm, t_scale, t_offset = scaleOffset(t_phys)

# write python code with scale information
with open(outputFileName, 'w') as f:
    f.write('def scales ():\n')
    f.write('   input_scale  = [' +
        str(y_scale) + ',' +
        str(x_scale) + ',' + 
        str(t_scale) + ',' + 
        str(z_scale) + ']\n')
    f.write('   input_offset = [' +
        str(y_offset) + ',' +
        str(x_offset) + ',' +
        str(t_offset) + ',' +
        str(z_offset) + ']\n')
    f.write('   output_scale = [' +
        str(1.0/u_scale) + ',' +
        str(1.0/u_scale) + ',' +
        str(1.0/u_scale) + ',' +
        str(1.0/lr_scale) + ',' +
        str(1.0/lt_scale) + ']\n')
    f.write('   output_offset = [0,0,0,' +
        str(-lr_offset/lr_scale) + ',' +
        str(-lt_offset/lt_scale) + ']\n')
    f.write('   return input_scale, input_offset, output_scale, output_offset\n')
    f.close()