# rhpinn: Radiative Hydrodynamics Physics Informed Neural Network

This repository contains the Python code to reproduce the results in the Keller (2025) paper that can be found [here](https://iopscience.iop.org/article/10.3847/1538-4357/add6a9/meta).

## Requirements

- `numpy`
- `astropy`
- `matplotlib`
- `TensorFlow`


## Opacity deep neural networks

Follow the instructions in the Opacity subfolder to create the `Opacity/opacity_rosseland.keras` and `Opacity/opacity_500nm.keras` deep neural networks that output the opacity based on temperature and pressure.


## Bifrost simulation data

The Bifrost data qs024048_by3363 quiet sun simulation can be downloaded from sdc.uio.no/search/simulations. Download

    BIFROST_qs024048_by3363_lgr_850.fits
    BIFROST_qs024048_by3363_lgtg_850.fits
    BIFROST_qs024048_by3363_ux_850.fits
    BIFROST_qs024048_by3363_uy_850.fits
    BIFROST_qs024048_by3363_uz_850.fits

and the corresponding files for time steps 855, 860, ..., 950.

The python code to read the Bifrost data is in `readBifrost.py`. Change the directory information on line 25 to the directory where you downloaded the Bifrost data to.


## Bifrost subset

`extract.py` contains constants that define the range of Bifrost data to use in x,y,z and t. The current values correspond to what has been used in the paper.

## Neural-network scaling values

Neural networks work best when the input and output values are in the range of -1 to +1. Run

    python scaling.py

to create the file `scales.py`. This fie can then be imported by the other codes to always work with the same part of the Bifrost simulations.


## Deep Neural Network trained directly on Bifrost simulation

To check that the deep neural network architecture in `dnn.py` can, in principle, represent a Bifrost simulation in great detail, we can fit the neural network directly to the Bifrost model with

    python bifnn.py

With the default parameters, a loss of better than 6e-4 should be achieved in 100 epochs. Note that the default batch size and the learning rate were optimized for the M3Max processor with 64GB. A different hardware configuration may converge faster with a different choice of batch size and learning rate. This can be achieved by changing the `bifnn.py` command-line arguments.

    usage: bifnn [-h] [-y DELTAY] [-s STEP] [-e EPOCHS] [-i INPUT_MODEL]
                [-o OUTPUT_MODEL] [-l LEARNING_RATE] [-b BATCH_SIZE]

    Train a deep neural network to reproduce a Bifrost simulation

    options:
    -h, --help              show this help message and exit
    -y DELTAY, --deltaY DELTAY
                            change in y extraction
    -s STEP, --step STEP  step when comparing to data in x,y,z
    -e EPOCHS, --epochs EPOCHS
                            number of epochs
    -i INPUT_MODEL, --input_model INPUT_MODEL
                            model to start from
    -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                            neural network model name
    -l LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning rate
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size


## int2mod deep neural network

The `int2mod` deep neural network takes a normalized intensity value and estimates the temperature, density and vertical velocity stratifications. It needs to be trained on a section of the Bifrost simulations that is different from the location where we want to validate the PINN.


### Deep Neural Network at different location in Bifrost simulation

The first step towards the `int2mod` model is the creation of another deep neural network that is trained directly on the Bifrost simulations, but now in a different location. The command line argument `-y` can be used to change the location where the model is trained. Here we chose an area 100 grid points in the positive y direction.

    python bifnn.py -y 100 -o bifnn_y100.keras


### Images from Bifrost model in different area

Using the `bifnn_y100.keras` model, we can now create the corresponding continuum images as input for the training of the `int2mod` model.

    python contimag.py -i bifnn_y100.keras -d imag_bifrost_y100

`contimag.py` has the following command line arguments

    usage: contimag [-h] [-i INPUT_MODEL] [-o OPACITY] [-d DIRECTORY_IMAGES]

    Calculate images from deep neural network hydrodynamics models

    options:
    -h, --help            show this help message and exit
    -i INPUT_MODEL, --input_model INPUT_MODEL
                            neural-network model
    -o OPACITY, --opacity OPACITY
                            opacity neural-network
    -d DIRECTORY_IMAGES, --directory_images DIRECTORY_IMAGES
                            directory where images are written to


### int2mod training

    python int2mod.py -y 100 -i imag_bifrost_y100

With the default parameters, a loss of about 0.06 should be achieved in 200 epochs.

`int2mod.py` has the following command line arguments

    usage: int2mod [-h] [-y DELTAY] [-i IMAGE_DIRECTORY] [-o OUTPUT_MODEL]
                [-e EPOCHS] [-l LEARNING_RATE] [-b BATCH_SIZE]

    Train a neural network to provide physical parameters based on intensity
    images alone

    options:
    -h, --help              show this help message and exit
    -y DELTAY, --deltaY DELTAY
                            change in y extraction
    -i IMAGE_DIRECTORY, --image_directory IMAGE_DIRECTORY
                            image directory
    -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                            neural network output model name
    -e EPOCHS, --epochs EPOCHS
                            number of epochs
    -l LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning rate
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size


## Neural network reproducing average density and temperature stratifications

The PINN will be trained to match the average density and temperature stratification. Since the z-axis has irregular spacing, it is easiest to train a neural network to provide an analytical function that outputs density and temperature as a function of height. This can be achieved by

    python strat.py

which saves the neural network in `strat.keras`.


## Images from Bifrost model
To create the continuum images at 500 nm, we again use `contimag.py`:

    python contimag.py -i bifnn.keras


## Initial PINN with int2mod

The first step in creating the PINN is to train a first version using the `int2mod` neural network to create a first atmospheric model pixel by pixel in the images while using an approximation of the continuity equation to estimate the horizontal velocities: 

    python rhstart.py -m imag_bifrost -n int2mod.keras

A loss of about 3e-4 should be reached within 25 epochs using the default parameters. These can be changed via command-line arguments:

    usage: rhstart [-h] [-m IMAGE_DIRECTORY] [-s STEP] [-i INPUT_MODEL]
                [-n INT2MOD] [-o OUTPUT_MODEL] [-e EPOCHS] [-l LEARNING_RATE]
                [-b BATCH_SIZE]

    Initial PINN training using int2mod Deep Neural Network

    options:
    -h, --help            show this help message and exit
    -m IMAGE_DIRECTORY, --image_directory IMAGE_DIRECTORY
                            image directory
    -s STEP, --step STEP  step when comparing to data in x,y,z
    -i INPUT_MODEL, --input_model INPUT_MODEL
                            model to start from
    -n INT2MOD, --int2mod INT2MOD
                            int2mod deep neural network
    -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                            neural network output model name
    -e EPOCHS, --epochs EPOCHS
                            number of epochs
    -l LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning rate
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size


## Coarse PINN training

It is advantageous to first train the PINN on a coarser scale in x,y and t. The `-s` argument allows us to define the step size in those three axes. For simplicity, we use a factor of 2, which already accelerates the training considerably.

    python rhmod.py -s 2 -m imag_bifrost -o rhmod_s2.keras

With the default learning rate and 50 epochs, a loss of better than 0.04 should be reached. Other parameters can be adjusted with command-line arguments

    usage: rhmod [-h] [-m IMAGE_DIRECTORY] [-z STRATIFICATION] [-s STEP]
                [-i INPUT_MODEL] [-o OUTPUT_MODEL] [-e EPOCHS] [-l LEARNING_RATE]
                [-b BATCH_SIZE]

    PINN training using full physics constraints

    options:
    -h, --help              show this help message and exit
    -m IMAGE_DIRECTORY, --image_directory IMAGE_DIRECTORY
                            image directory
    -z STRATIFICATION, --stratification STRATIFICATION
                            mean stratification model
    -s STEP, --step STEP  step when comparing to data in x,y,z
    -i INPUT_MODEL, --input_model INPUT_MODEL
                            model to start from
    -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                            neural network output model name
    -e EPOCHS, --epochs EPOCHS
                            number of epochs
    -l LEARNING_RATE, --learning_rate LEARNING_RATE
                            learning rate
    -b BATCH_SIZE, --batch_size BATCH_SIZE
                            batch size


## Fine PINN training

In the second training step, we start with the PINN trained on a coarse grid and train it on the full grid of points where the physics constraints are enforced

    python rhmod.py -m imag_bifrost -i rhmod_s2.keras -o rhmod_s1.keras -b 102400 -l 0.00002 -e 20


## Citing

If you use parts or all of the code in this repository in their original or modified forms for your own research, please cite Keller (2025).

    @ARTICLE{2025ApJ...989...92K,
       author = {{Keller}, Christoph U.},
        title = "{Data-driven Radiative Hydrodynamics Simulations of the Solar Photosphere Using Physics-informed Neural Networks: Proof of Concept}",
      journal = {\apj},
     keywords = {Solar physics, Hydrodynamical simulations, Astronomy data reduction, Neural networks, 1476, 767, 1861, 1933, Solar and Stellar Astrophysics},
         year = 2025,
        month = aug,
       volume = {989},
       number = {1},
          eid = {92},
        pages = {92},
          doi = {10.3847/1538-4357/add6a9},
    archivePrefix = {arXiv},
       eprint = {2505.04865},
    primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...989...92K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }




## Contributions

This repository serves as a way for interested parties to access the source code and replicate the results in the paper. There is no doubt that the code itself could be substantially improved, and it most certainly contains errors. As the current code serves as a proof of concept, there are no plans to do more than correct errors that influence the results.

If you find an error that influences the results in a significant way, then please let me know through the Github issue system.
