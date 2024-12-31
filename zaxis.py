# Irregular z-axis sampling to focus on region where energy goes from internal to radiation 
# author: Christoph U.Keller, ckeller@nso.edu

from scales import scales # coordinate and physical parameter scales
import numpy as np

def make_zaxis():
    # adjustable parameters
    nz = 64  # grid points in z direction

    # determine scales over which PINN is valid
    input_scale, input_offset, output_scale, output_offset = scales()
    zscale = 2.0 / input_scale[3]
    zoffset = (1 + input_offset[3]) / input_scale[3]

    # manual array of sampling differences
    xp = np.array([ 0, 5, 10, 20, 50, 57, 63]) # z-index 
    fp = np.array([50, 30, 20, 10, 10, 20, 30]) # desired step-size in km at z-indices
    zstep = np.interp(range(0,nz),xp,fp) * 1000.0 

    # normalize z step size so that it spans the original zmin to zmax
    zstep = zstep * zscale / np.sum(zstep)

    # integrate zstep to obtain intervals
    zaxis = np.cumsum(zstep) - zoffset

    return zaxis
