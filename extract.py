# defines extraction range and steps in BiFrost models

xstart = 404
ystart = 130
zstart = 40
tstart = 850

xstop = xstart + 96
ystop = ystart + 96
zstop = zstart + 64
tstop = tstart + 100

xstep = 1 # step when extracting data in x
ystep = 1 # step when extracting data in y
zstep = 1 # step when extracting data in z
tstep = 5 # step when extracting data in t

# range in x,y,z axes: start, stop, step
xrange = slice(xstart, xstop, xstep)
yrange = slice(ystart, ystop, ystep)
zrange = slice(zstart, zstop, zstep)
trange = slice(tstart, tstop, tstep)
