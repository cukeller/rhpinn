# calculate physics losses for radiative hydrodynamics PINN
# author: Christoph U.Keller, ckeller@nso.edu

from saha_tf import saha
import math
import tensorflow as tf

# === constants ===
ndim =  4 # number of dimensions (x,y,z,t)
gsun =  274.0 # solar surface gravity in m/s^2
sigma = 5.67e-8 # Stefan-Boltzmann constant in W/m^2/K^4
eVtoJ = 1.602176634e-19 # 1eV in Joule
amu =   1.660539e-27 # atomic mass unit
mmw =   1.29 # mean molecular weight of neutral solar composition gas in amu


def integrate(y, x, axis=-1, reverse=False, cummulative = False):
    length = y.shape[axis]
    if (reverse):
        index0 = tf.range(length - 1, 0, -1)
        index1 = tf.range(length, 1, -1)
        dx = tf.gather(x, index0, axis=axis) - tf.gather(x, index1, axis=axis)
    else:
        index0 = tf.range(0, length - 1)
        index1 = tf.range(1, length)
        dx = tf.gather(x, index1, axis=axis) - tf.gather(x, index0, axis=axis)

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
        return 0.5 * tf.reduce_sum(
            (tf.gather(y, index1, axis=axis) + tf.gather(y, index0, axis=axis)) * dx,
            axis=axis
        )

# load Rosseland neural network
rosseland = tf.keras.models.load_model('Opacity/opacity_rosseland.keras', compile=False)
# load continuum opacity neural network
contopac = tf.keras.models.load_model('Opacity/opacity_500nm.keras', compile=False)


# physics losses
@tf.function
def loss_physics(y, x, t, z, model, nz, nt, zaxis, strat):
    '''coordinates are in physical units, i.e. meters'''

    with tf.GradientTape(persistent = True) as g:
        g.watch([x,y,z,t])

        Xf = tf.reshape(tf.stack([y, x, t, z], axis = -1),[-1, ndim])
        Y = model(Xf)

        # extract estimated physical quantities
        # DNN uses 10log to match Bifrost, but PDEs work with ln
        vx =    Y[:,0]
        vy =    Y[:,1]
        vz =    Y[:,2]
        lnrho = Y[:,3] * math.log(10)
        lt =    Y[:,4]

        rho = tf.exp(lnrho)

        # hydrogen ionization fraction
        ionfrac = saha(tf.pow(10.0,lt), rho)

        # ideal gas law for solar photosphere mean molecular weight
        lnp = lnrho + lt * math.log(10) + 8.7711096 + tf.math.log(1.0 + ionfrac * 0.934)
        pgas = tf.exp(lnp)

        # internal energy of gas
        e = 3.0/2.0 * pgas / rho + ionfrac * 0.934 / (amu * 1.29) * 13.6 * eVtoJ

    # these lines can be outside of GradientTape as no derivatives are involved

    # opacity, rosseland() is in CGS units
    kappa = tf.pow(10.0, rosseland(tf.stack((lt, lnp / math.log(10) + 1), axis=1))[:,0] - 1)
    # continuum opacity
    kappac = tf.pow(10.0, contopac(tf.stack((lt, lnp / math.log(10) + 1), axis=1))[:,0] - 1)


    # calculate derivatives
    [dvx_dx, dvx_dy, dvx_dz, dvx_dt] = g.gradient(vx,    [x, y, z, t])
    [dvy_dx, dvy_dy, dvy_dz, dvy_dt] = g.gradient(vy,    [x, y, z, t])
    [dvz_dx, dvz_dy, dvz_dz, dvz_dt] = g.gradient(vz,    [x, y, z, t])
    [dlr_dx, dlr_dy, dlr_dz, dlr_dt] = g.gradient(lnrho, [x, y, z, t])
    [dlp_dx, dlp_dy, dlp_dz]         = g.gradient(lnp,   [x, y, z])
    [de_dx,  de_dy,  de_dz,  de_dt]  = g.gradient(e,     [x, y, z, t])

    # continuity equation
    loss_continuity = tf.reduce_mean(tf.square(
        dlr_dt + # comment out for anelastic approximation
        vx*dlr_dx + vy*dlr_dy + vz*dlr_dz + dvx_dx + dvy_dy + dvz_dz
    )) / tf.math.reduce_variance(vz)

    # momentum equations in x, y and z
    lmx = tf.reduce_mean(tf.square(
        dvx_dt + vx*dvx_dx + vy*dvx_dy + vz*dvx_dz + pgas/rho * dlp_dx
    )) / 1.0e6

    lmy = tf.reduce_mean(tf.square(
        dvy_dt + vx*dvy_dx + vy*dvy_dy + vz*dvy_dz + pgas/rho * dlp_dy
    )) / 1.0e6

    lmz = tf.reduce_mean(tf.square(
        # + gsun because gravity vector goes in negative z direction
        dvz_dt + vx*dvz_dx + vy*dvz_dy + vz*dvz_dz + pgas/rho * dlp_dz + gsun
    )) / tf.math.reduce_variance(vz)

    tau = integrate(
        tf.reshape(kappa*rho, [-1, nz]),
        zaxis,
        reverse=True,
        cummulative=True
    )

    tau1d = tf.reshape(tau,[-1]) # flatten to 1-D
    # qrad is per unit mass
    qrad = (0.3366811 * sigma * kappa * tf.pow(10.0, 4.0 * lt) * 
            tf.exp(-tau1d * 0.0178391) / (1.0 + tau1d * 4.4409090))

    # internal energy equation
    loss_energy = tf.reduce_mean(tf.square(
        de_dt + 
        vx*de_dx + vy*de_dy + vz*de_dz + 
        pgas/rho * (dvx_dx + dvy_dy + dvz_dz) + qrad
    ))

    # stratification losses for log10(rho) at all times
    loss_mlr = tf.reduce_mean(tf.square(
        tf.reduce_mean(tf.reshape(Y[:,3], [-1,nt,nz]), axis=0) - strat[:,0]
    ))

    # stratification losses for log10(T) at all times
    loss_mlt = tf.reduce_mean(tf.square(
        tf.reduce_mean(tf.reshape(Y[:,4], [-1,nt,nz]), axis=0) - strat[:,1]
    ))

    # no net mass flow at any time or height
    # weighed by 1/mean(rho) since mean(vz*rho) goes like ~exp(-z)
    loss_mflux = tf.reduce_mean(tf.square(
        tf.reduce_mean(tf.reshape(vz*rho,[-1,nt,nz]), axis=0) / 
        tf.reduce_mean(tf.reshape(   rho,[-1,nt,nz]), axis=0)
    ))
    
    # energy flux as a function of z
    massflux = tf.reshape(vz*rho,[-1,nt,nz])
    netmassflux = massflux - tf.reduce_mean(massflux, axis=0, keepdims = True)
    enerflux = tf.reduce_mean(netmassflux * tf.reshape(e  + pgas/rho + 0.5 * (vx**2+vy**2+vz**2), [-1, nt, nz]), axis=0)

    # cummulative radiative loss = radiative energy flux as a function of z
    radflux = tf.reduce_mean(integrate(
        tf.reshape(qrad*rho, [-1,nt,nz]), zaxis, axis=-1, cummulative=True), axis=0)
    loss_eflux = tf.reduce_mean(tf.square(enerflux + radflux - 6.3e7)) # - 5.8e7))

    # formal solution of the radiative transfer equation for continuum images
    tauc = integrate(
        tf.reshape(kappac*rho, [-1, nz]),
        zaxis,
        reverse=True,
        cummulative=True
    )
    c =   1.191066E-5 # 2*h*c^2
    hck = 1.438832334 # h*c/k
    wl =  500e-7 # wavelength in cm
    # formal solution of continuum RTE
    intens = integrate(
        c/wl**5/tf.math.expm1(hck/wl/tf.reshape(tf.pow(10.0, lt), [-1,nz])) * tf.exp(-tauc),
        tauc,
        reverse=True
    ) / 5e14

    return tf.stack([loss_continuity, lmx, lmy, lmz, loss_energy, \
        loss_mflux, loss_eflux, loss_mlr, loss_mlt]), intens
