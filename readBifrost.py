# read Bifrost simulation data
# author: Christoph U.Keller, ckeller@nso.edu

import numpy as np
from astropy.io import fits

def axisLength (slice):
    return (slice.stop - slice.start) // slice.step


def readBifrost (params, extr, step=1, read=True):
    xr = slice(extr.xstart, extr.xstop, extr.xstep * step)
    yr = slice(extr.ystart, extr.ystop, extr.ystep * step)
    zr = slice(extr.zstart, extr.zstop, extr.zstep * step)
    tr = slice(extr.tstart, extr.tstop, extr.tstep * step)

    # number of grid points in each axis
    nx = axisLength(xr)
    ny = axisLength(yr)
    nz = axisLength(zr)
    nt = axisLength(tr)

    # simulation data directory and first part of file name
    dir = '../BiFrost Simulation/qs024048_by3363/atmos/'
    prefix = 'BIFROST_qs024048_by3363_'

    # read non-uniform z-axis from FITS extension
    z = fits.getdata(dir + prefix + 'uz_850.fits',1)[zr]

    # get coordinate scales in simulation data
    header = fits.getheader(dir + prefix + 'uz_850.fits')
    xscale = header['CDELT1'] * xr.step * 1.0e6 # x increment in m
    yscale = header['CDELT2'] * yr.step * 1.0e6 # y increment in m
    zscale = 1.0e6 # z scale in m; scale itself is not uniform
    tscale = 10 * tr.step # time increment in seconds

    # parameter grid
    npar = len(params)
    par = np.zeros((nt,nz,ny,nx,npar), dtype=np.float32)

    if (read):
        # read all data into respective arrays
        for it in range(tr.start,tr.stop,tr.step):
            ts = str(it)
            tindex = int((it - tr.start)/tr.step)
            for ip in range(0,len(params)):
                par[tindex,:,:,:,ip] = fits.getdata(
                    dir + prefix + params[ip]+'_' +ts+'.fits'
                )[zr,yr,xr]

    xaxis = np.arange(0, nx) * xscale
    yaxis = np.arange(0, ny) * yscale
    zaxis = z * zscale
    taxis = np.arange(0, nt) * tscale

    return par, (xaxis, yaxis, zaxis, taxis), (xscale, yscale, zscale, tscale)
