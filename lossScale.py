# loss scaling for radiative hydrodynamics PINN
# author: Christoph U.Keller, ckeller@nso.edu

loss_scale = [
    1e0,   # loss_image;
    1e9,   # loss_continuity; 1e9
    1e1,   # loss momentum x; 1e1
    1e1,   # loss momentum y; 1e1
    1e1,   # loss momentum z; 1e1
    1e-14, # loss_energy
    1e-8,  # loss_mflux; 1e-7
    1e-17, # loss_eflux; 1e-18
    1e0,   # loss_mlr
    1e0,   # loss_mlt; was 1e1
]

loss_names = [
    'image       ',
    'continuity  ',
    'momentum_x  ',
    'momentum_y  ',
    'momentum_z  ',
    'energy      ',
    'mass_flux   ',
    'energy_flux ',
    'log(rho)(z) ',
    'log(T)(z)   ',
]